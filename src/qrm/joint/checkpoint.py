from contextlib import contextmanager, nullcontext
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import re
import shutil
import time
import numpy as np
import torch

from qrm.checkpoint_retention import keep_checkpoint, validate_retention


def clone_cpu_state(state, aliases=None):
    """Own the tensors without duplicating tied weights in every checkpoint.

    Only identical storage views are reused; distinct slices remain distinct.
    Keeping aliases is lossless and torch.save serializes their storage once.
    """
    copies, result = {}, {}
    for name in state:
        value = state[(aliases or {}).get(name, name)]
        key = (value.device, value.dtype, value.untyped_storage().data_ptr(),
               value.storage_offset(), tuple(value.shape), tuple(value.stride()))
        if key not in copies:
            copies[key] = value.detach().cpu().clone()
        result[name] = copies[key]
    return result


class ShardEMA:
    """FP32 EMA of local original-parameter shards; updated once per optimizer step."""
    def __init__(self, model, decay=.999):
        self.decay = decay
        self.values = {n: p.detach().float().clone() for n,p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n,p in model.named_parameters():
            if n in self.values:
                self.values[n].lerp_(p.detach().float(), 1-self.decay)

    @contextmanager
    def applied(self, model):
        backup = {}
        with torch.no_grad():
            for n,p in model.named_parameters():
                if n in self.values:
                    backup[n] = p.detach().clone()
                    p.copy_(self.values[n])
        try:
            yield
        finally:
            with torch.no_grad():
                for n,p in model.named_parameters():
                    if n in backup:
                        p.copy_(backup[n])


def capture_rng():
    numpy_state = np.random.get_state()
    return dict(cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                python=random.getstate(),
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]))


def optimizer_layout(model, optimizer):
    names = {id(p): n for n,p in model.named_parameters()}
    return [[(names[id(p)], tuple(p.shape), str(p.dtype)) for p in group['params']]
            for group in optimizer.param_groups]


def restore_rng(state):
    torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        name, keys, position, has_gauss, cached = state['numpy']
        np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))


def complete_checkpoint(directory):
    directory = Path(directory)
    if directory.is_symlink() or not re.fullmatch(r'step_\d{6,}', directory.name):
        return False
    try:
        marker = json.loads((directory/'complete.json').read_text())
        if marker['step'] != int(directory.name[5:]):
            return False
        for key in ('ema', 'trainer'):
            if marker[key] != key+'.pt':
                return False
            file = directory/marker[key]
            if not file.is_file() or file.stat().st_size <= 0:
                return False
            if marker.get('sizes', {}).get(key, file.stat().st_size) != file.stat().st_size:
                return False
        if marker.get('optimizer_format') == 'rank_local_v1':
            for rank in range(marker['world_size']):
                key = f'optimizer_rank_{rank:04d}.pt'
                if (directory/key).stat().st_size != marker['optimizer_sizes'][key]:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def latest_complete_checkpoint(out):
    candidates = [p for p in (Path(out)/'checkpoints').glob('step_*') if complete_checkpoint(p)]
    if not candidates:
        raise FileNotFoundError('No complete checkpoint to resume in this output directory')
    return max(candidates, key=lambda p: int(p.name[5:]))


def retention_settings(out, training):
    """Runtime retention overrides do not change strict resume provenance."""
    path = Path(out)/'checkpoint_retention_policy.json'
    if not path.exists():
        return dict(training)
    override = json.loads(path.read_text())
    if override.get('schema_version') != 1 or override.get('checkpoint_retention') != 'all':
        raise ValueError('Unsupported runtime checkpoint retention policy')
    save = override['save_every']
    keep = override['keep_every']
    if not isinstance(save, int) or not isinstance(keep, int) or save <= 0 or keep != save:
        raise ValueError('Permanent checkpoint interval must equal the save interval')
    if training.get('save_every', save) != save:
        raise ValueError('Runtime retention must not change checkpoint save frequency')
    return dict(training, checkpoint_retention='all', keep_every=keep)


def preserve_complete_checkpoint(source, permanent):
    """Pin durable checkpoint files with hard links, then publish atomically."""
    source, permanent = Path(source), Path(permanent)
    if not complete_checkpoint(source):
        return False
    permanent.mkdir(parents=True, exist_ok=True)
    destination = permanent/source.name
    if destination.exists():
        if not complete_checkpoint(destination):
            raise ValueError(f'Incomplete permanent checkpoint: {destination}')
        return False
    marker = json.loads((source/'complete.json').read_text())
    names = ['ema.pt', 'trainer.pt']
    if marker.get('optimizer_format') == 'rank_local_v1':
        names += [f'optimizer_rank_{rank:04d}.pt' for rank in range(marker['world_size'])]
    staging = permanent/f'.preserving_{os.getpid()}_{time.time_ns()}'
    snapshot = staging/source.name
    snapshot.mkdir(parents=True)
    try:
        for name in [*names, 'complete.json']:
            os.link(source/name, snapshot/name)
        if not complete_checkpoint(snapshot):
            raise ValueError('Checkpoint changed while preserving it')
        descriptor = os.open(snapshot, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        snapshot.rename(destination)
        descriptor = os.open(permanent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        shutil.rmtree(staging)
    return True


def prune_joint_checkpoints(out, latest, training):
    """Delete temporary snapshots only after the replacement is durable.

    Legacy milestone runs protect selected best checkpoints. The history policy
    keeps only the latest before keep_from, including when an older point was best.
    This function never follows checkpoint-directory symlinks or visits other runs.
    """
    training = retention_settings(out, training)
    validate_retention(training)
    if training.get('checkpoint_retention', 'all') == 'all':
        return []
    out, latest = Path(out), Path(latest)
    if not complete_checkpoint(latest) or latest.parent.resolve() != (out/'checkpoints').resolve():
        raise ValueError('Refusing cleanup without a complete replacement in this run')
    protected = {latest.resolve()}
    protect_best = training['checkpoint_retention'] == 'milestones_and_latest'
    if protect_best:
        protected.add((out/'best').resolve())
    selected = out/'best_validation.json'
    selection = json.loads(selected.read_text()) if selected.exists() else None
    if protect_best and selection and selection.get('checkpoint'):
        protected.add(Path(selection['checkpoint']).resolve().parent)
    removed = []
    for candidate in sorted((out/'checkpoints').glob('step_*')):
        if not complete_checkpoint(candidate) or candidate.resolve() in protected:
            continue
        step = int(candidate.name[5:])
        if keep_checkpoint(step, int(latest.name[5:]), training):
            continue
        # New latest-only policy takes precedence over early best selections.
        # Preserve the historical score, but never leave a dangling best link.
        best = out/'best'
        if best.is_symlink() and best.resolve() == candidate.resolve():
            best.unlink()
        if selection and selection.get('checkpoint') and Path(selection['checkpoint']).resolve().parent == candidate.resolve():
            selection.update(checkpoint_available=False, checkpoint_pruned_by_retention=True)
            temporary = selected.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(selection, indent=2)+'\n')
            temporary.replace(selected)
        shutil.rmtree(candidate)
        removed.append(step)
    if removed:
        with (out/'checkpoint_retention.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(latest=int(latest.name[5:]), removed_temporary_steps=removed))+'\n')
    return removed


def durable_save(payload, file):
    temporary = file.with_suffix(file.suffix+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(file)


class StagedDurability:
    """Serialize synchronously; collect every file-stage failure before commit.

    Background workers only fsync/close/rename;
    they never serialize live model/Adam tensors or touch CUDA. A failed open,
    save, flush, cleanup close or submit becomes an error returned by finish,
    so each rank still reaches the checkpoint error gather/broadcast.
    """
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = []
        self.errors = []

    @staticmethod
    def _finish_file(stream, temporary, file):
        error = None
        try:
            os.fsync(stream.fileno())
        except BaseException as exc:
            error = exc
        try:
            stream.close()
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None:
            raise error
        temporary.replace(file)

    def save(self, payload, file):
        stream = None
        try:
            temporary = file.with_suffix(file.suffix + '.tmp')
            stream = temporary.open('wb')
            torch.save(payload, stream)
            stream.flush()
            future = self.executor.submit(self._finish_file, stream, temporary, file)
        except BaseException as error:
            self.errors.append(error)
            if stream is not None:
                try:
                    stream.close()
                except BaseException as close_error:
                    self.errors.append(close_error)
            return
        # The worker owns the stream only after successful submit.
        self.pending.append(future)

    def finish(self):
        try:
            self.executor.shutdown(wait=True)
        except BaseException as error:
            self.errors.append(error)
        for future in self.pending:
            try:
                future.result()
            except BaseException as error:
                self.errors.append(error)
        if self.errors:
            raise self.errors[0]


def save_joint_checkpoint(model, optimizer, ema, distributed, out, step, metadata):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    sharded = isinstance(model, FSDP)
    # FSDP may clone tied entries separately before returning its full state.
    # Recover their identity from the actual Parameter objects, never by value.
    names, aliases = {}, {}
    for name, param in model.named_parameters(remove_duplicate=False):
        name = name.replace('_fsdp_wrapped_module.', '')
        if id(param) in names:
            aliases[name] = names[id(param)]
        else:
            names[id(param)] = name
    def full_state():
        offload = next(model.parameters()).device.type != 'cpu'
        context = (FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                   FullStateDictConfig(offload_to_cpu=offload, rank0_only=True)) if sharded else nullcontext())
        with context:
            state = model.state_dict()
            return clone_cpu_state(state, aliases) if distributed.primary else None
    raw = full_state()
    with ema.applied(model):
        averaged = full_state()
    rng = distributed.gather(capture_rng())
    directory = Path(out)/'checkpoints'/f'step_{step:06d}'
    if distributed.primary:
        if directory.exists() and not complete_checkpoint(directory):
            # Preserve evidence from interrupted writes; never mistake it for a
            # usable checkpoint and never let it prevent saving the replayed step.
            directory.rename(directory.with_name(directory.name+f'.incomplete_{time.time_ns()}'))
        directory.mkdir(parents=True, exist_ok=False)
    distributed.barrier()
    # Save Adam exactly as owned by this rank. This avoids a full optimizer
    # all-gather and PyTorch 2.7 CPU full-state export's reused moment buffer.
    # Same-world-size resume validates every parameter name/shape and group.
    durability = StagedDurability()
    optimizer_file = directory/f'optimizer_rank_{distributed.rank:04d}.pt'
    durability.save(dict(step=step, rank=distributed.rank, world_size=distributed.world_size,
                      layout=optimizer_layout(model, optimizer), optimizer=optimizer.state_dict()), optimizer_file)
    distributed.barrier()
    if distributed.primary:
        schema = metadata.get('config', {}).get('schema_version', 2)
        stage = 'history_sft' if schema in (3, 4) else 'joint_sft'
        payload = dict(metadata, schema_version=schema, stage=stage, step=step, weights='complete_ema', model=averaged)
        durability.save(payload, directory/'ema.pt')
        durability.save(dict(metadata, schema_version=schema, stage=stage, step=step, model=raw,
                          optimizer_format='rank_local_v1', rng=rng,
                          next_global_sample=step*metadata['config']['sft']['global_batch_size']), directory/'trainer.pt')
    # Each rank reports flush errors before publishing any completeness marker.
    failure = None
    try:
        durability.finish()
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
    failures = distributed.gather(failure)
    failure = distributed.broadcast(next((x for x in failures if x), None) if distributed.primary else None)
    if failure:
        raise RuntimeError('Checkpoint durability failed: ' + failure)
    if distributed.primary:
        marker = dict(step=step, ema='ema.pt', trainer='trainer.pt',
                      optimizer_format='rank_local_v1', world_size=distributed.world_size,
                      optimizer_sizes={f'optimizer_rank_{rank:04d}.pt':
                                       (directory/f'optimizer_rank_{rank:04d}.pt').stat().st_size
                                       for rank in range(distributed.world_size)},
                      sizes={key:(directory/(key+'.pt')).stat().st_size for key in ('ema','trainer')})
        with (directory/'complete.json.tmp').open('w') as stream:
            stream.write(json.dumps(marker)+'\n'); stream.flush(); os.fsync(stream.fileno())
        (directory/'complete.json.tmp').replace(directory/'complete.json')
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        # Publish the checkpoint directory entry durably before deleting the
        # previous restart point, not just its internal complete marker.
        descriptor = os.open(directory.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        update_link(Path(out)/'latest', directory)
        descriptor = os.open(out, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        prune_joint_checkpoints(out, directory, metadata['config'].get('training', {}))
    distributed.barrier()
    return directory/'ema.pt'


def restore_joint_checkpoint(model, optimizer, distributed, directory, expected):
    """Load full FP32 model/Adam/EMA state into the current FSDP shards.

    World size is intentionally fixed so each rank resumes its own RNG stream.
    Every rank reads CPU model tensors and only its own optimizer file.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
    directory = Path(directory)
    if not complete_checkpoint(directory):
        raise ValueError('Cannot resume an incomplete checkpoint')
    payload = torch.load(directory/'trainer.pt', map_location='cpu', weights_only=True)
    average = torch.load(directory/'ema.pt', map_location='cpu', weights_only=True)
    for key in ('config', 'method', 'seed', 'backbone_sha256', 'data_manifest_sha256',
                'plan_sha256', 'actual_world_size', 'actual_accumulation_steps', 'norm_stats_sha256',
                'history_index_sha256', 'memory_checkpoint_sha256', 'initial_checkpoint_sha256',
                'history_stage', 'source_signature', 'action_prerequisites', 'initialization',
                'automatic_validation'):
        if key in expected and (payload.get(key) != expected[key] or average.get(key) != expected[key]):
            raise ValueError(f'Resume configuration/provenance mismatch: {key}')
    step = payload['step']
    if (average['step'] != step or average.get('weights') != 'complete_ema'
            or step != int(directory.name[5:])
            or payload['next_global_sample'] != step*expected['config']['sft']['global_batch_size']
            or len(payload['rng']) != distributed.world_size):
        raise ValueError('Resume step, EMA, sampler or rank RNG state mismatch')
    sharded = isinstance(model, FSDP)
    context = (FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
               FullStateDictConfig(offload_to_cpu=next(model.parameters()).device.type != 'cpu',
                                   rank0_only=False)) if sharded else nullcontext())
    with context:
        model.load_state_dict(average['model'], strict=True)
        ema = ShardEMA(model, expected['config']['sft'].get('ema_decay', .999))
        del average
        model.load_state_dict(payload.pop('model'), strict=True)
        if payload.get('optimizer_format') == 'rank_local_v1':
            local = torch.load(directory/f'optimizer_rank_{distributed.rank:04d}.pt',
                               map_location='cpu', weights_only=True)
            if (local['step'] != step or local['rank'] != distributed.rank
                    or local['world_size'] != distributed.world_size
                    or local['layout'] != optimizer_layout(model, optimizer)):
                raise ValueError('Optimizer rank, parameter layout or step mismatch')
            optimizer_state = local['optimizer']
        else:
            optimizer_state = payload.pop('optimizer')
            if sharded:
                optimizer_state = FSDP.optim_state_dict_to_load(model, optimizer, optimizer_state)
        optimizer.load_state_dict(optimizer_state)
        # FSDP's full optimizer export can reuse the same scalar step storage
        # across original parameters. AdamW increments each parameter's step
        # in-place; sharing it changes bias correction after a restart.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    state[key] = value.clone()
    restore_rng(payload['rng'][distributed.rank])
    return step, ema


def update_link(link, target):
    link, target = Path(link), Path(target)
    temporary = link.with_name(link.name+'.tmp')
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target.resolve(), target_is_directory=target.is_dir())
    temporary.replace(link)


def select_best(out, checkpoint, macro_tsr, step):
    path = Path(out)/'best_validation.json'
    old = json.loads(path.read_text()) if path.exists() else None
    if old is None or macro_tsr > old['macro_TSR']:
        path.write_text(json.dumps(dict(checkpoint=str(Path(checkpoint).resolve()), macro_TSR=macro_tsr, step=step), indent=2)+'\n')
        update_link(Path(out)/'best', Path(checkpoint).parent)
        return True
    return False
