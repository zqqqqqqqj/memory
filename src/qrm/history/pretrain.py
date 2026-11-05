"""Stage one: supervise causal residual memory from prepared fixed visual features."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
import time
import uuid

import torch

from qrm.config import ROOT, load_config
from qrm.checkpoint_retention import keep_checkpoint, validate_retention
from qrm.distributed import DistributedContext, loss_rng
from qrm.history.config import architecture_signature, build_memory
from qrm.history.data import FeatureDataset, file_sha256, load_prefix, load_teacher_initialization
from qrm.history.input_contract import validate_input_contract
from qrm.history.tracking import TrainingTracker
from qrm.memory.stream_codec import MemoryState


def configure_stage(memory, stage, reader_mode='full'):
    if stage != 'memory':
        raise ValueError('Only the memory pretraining stage is supported by schema 4')
    memory.requires_grad_(False)
    for module in memory.history_train_modules(reader_mode):
        module.requires_grad_(True)
    memory.teacher.eval()
    parameters = [parameter for parameter in memory.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError('Memory objective has no trainable parameters')
    return parameters


def memory_objective(memory, record, end_index, *, device='cpu', max_queries=32,
                     generator=None, diagnostics=True, reader_mode="full", dataset=None):
    if not 0 <= end_index < len(record['times']):
        raise ValueError('Historical prefix endpoint is outside this episode')
    state = MemoryState()
    times = record['times'][:end_index+1]
    # A compact prefix is read once; no full visual tokens, target encoder, or
    # truncated writer graph participate in historical supervision.
    loader = dataset.load_prefix if dataset is not None else load_prefix
    features = loader(record, end_index, memory.teacher, device=device)
    for time, feature in zip(times, features, strict=True):
        state = memory.codec.append(state, feature, time)
    loss, metrics = memory.history_loss(state, times, features, max_queries=max_queries,
                                        generator=generator, diagnostics=diagnostics, mode=reader_mode)
    metrics['prefix_decisions'] = len(times)
    return loss, metrics


def _mean_metrics(rows, *, relative_energy_floor=1e-12):
    """Aggregate sufficient statistics before computing errors and ratios.

    Undefined relative errors are emitted as JSON null, never NaN and never a
    successful zero. Diagnostic metrics without sufficient statistics remain
    means across finite observations.
    """
    keys = sorted({key for row in rows for key in row})
    result = {}
    for key in keys:
        values = [float(row[key]) for row in rows
                  if row.get(key) is not None and math.isfinite(float(row[key]))]
        result[key] = sum(values)/len(values) if values else None
    statistics = ('delta_sse', 'target_delta_energy', 'query_elements', 'history_queries',
                  'eligible_history_queries', 'observed_past_queries', 'low_energy_queries',
                  'history_sse', 'history_elements')
    for key in keys:
        if any(key == suffix or key.endswith('_'+suffix) for suffix in statistics):
            result[key] = sum(float(row.get(key, 0)) for row in rows)
    for prefix in ('', 'near_', 'mid_', 'far_'):
        elements = result.get(prefix+'query_elements', 0)
        if prefix+'delta_sse' in result:
            result[prefix+'delta_mse'] = result[prefix+'delta_sse']/elements if elements else None
        if prefix+'target_delta_energy' in result:
            energy = result[prefix+'target_delta_energy']
            valid = energy > relative_energy_floor*elements and elements > 0
            result[prefix+'relative_error_valid'] = int(valid)
            result[prefix+'relative_error'] = result[prefix+'delta_sse']/energy if valid else None
    if 'history_elements' in result:
        elements = result['history_elements']
        result['history_mse'] = result['history_sse']/elements if elements else None
    for prefix in ('', 'near_', 'mid_', 'far_'):
        if prefix+'low_energy_queries' in result:
            queries = result.get(prefix+'history_queries', 0)
            result[prefix+'low_energy_coverage'] = result[prefix+'low_energy_queries']/queries if queries else None
    if 'observed_past_queries' in result:
        observed = result['observed_past_queries']
        result['eligible_query_coverage'] = result.get('eligible_history_queries', 0)/observed if observed else None
    if 'stored_mean_baseline_mse' in keys:
        elements = result.get('query_elements', 0)
        result['stored_mean_baseline_mse'] = result['target_delta_energy']/elements if elements else None
    if 'zero_residual_mse' in keys:
        elements = sum(row.get('query_elements', 0) for row in rows if 'zero_residual_mse' in row)
        total = sum(row['zero_residual_mse']*row.get('query_elements', 0)
                    for row in rows if 'zero_residual_mse' in row)
        result['zero_residual_mse'] = total/elements if elements else None
    return result


@torch.no_grad()
def evaluate(memory, dataset, stage, *, episodes=52, seed=8675309, device='cpu', max_queries=32,
             reader_mode='full', min_prefix_decisions=None):
    if stage != 'memory':
        raise ValueError('Only memory pretraining is supported')
    if dataset.split != 'validation':
        raise ValueError('Validation requires held-out episodes')
    memory.eval()
    rng = random.Random(seed)
    query_rng = torch.Generator().manual_seed(seed)
    rows, identities = [], []
    minimum = _prefix_minimum(memory, min_prefix_decisions)
    tasks = sorted(dataset.by_task)
    for number in range(episodes):
        task = tasks[number % len(tasks)]
        candidates = [r for r in dataset.by_task[task] if len(r['times']) >= minimum]
        if not candidates:
            raise ValueError(f'No validation episode for task {task} contains a merged history')
        record = rng.choice(candidates)
        end = rng.randrange(minimum-1, len(record['times']))
        _, metrics = memory_objective(memory, record, end, device=device, max_queries=max_queries,
                                       generator=query_rng, diagnostics=True, reader_mode=reader_mode, dataset=dataset)
        rows.append(metrics)
        identities.append(dict(episode_id=record['episode_id'], task_id=record['task_id'],
                               seed=record['seed'], current_time=record['times'][end]))
    memory.train()
    memory.teacher.eval()
    return dict(split='validation', episodes=episodes, reader_mode=reader_mode,
                metrics=_mean_metrics(rows, relative_energy_floor=memory.relative_energy_floor),
                identities=identities, ablations=('Same-model input ablations; independent controls require separate --reader-mode runs. '
                           'Time mode retains merge metadata (including visual-dependent Ward boundaries/counts); '
                           'it is a metadata-only control, not a strict image-independent time baseline.'))


def provenance_for(index_path, cfg):
    return dict(index_sha256=file_sha256(index_path), architecture=architecture_signature(cfg),
                source_files={name: file_sha256(ROOT/'src/qrm/history'/name)
                              for name in ('core.py', 'teacher.py', 'supervision.py', 'pretrain.py', 'data.py',
                                           'compact_cache.py', 'config.py', 'initialization.py', 'input_contract.py', 'tracking.py',
                                           '../distributed.py', '../checkpoint_retention.py', '../memory/stream_codec.py')})


def _rng_state(memory, training_rng, query_rng):
    device = next(memory.parameters()).device
    return dict(training=training_rng.getstate(), queries=query_rng.get_state(),
                torch=torch.get_rng_state(),
                # Reading all visible devices would initialize other ranks' GPU
                # contexts. This process owns only its local CUDA generator.
                cuda=None, cuda_local=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def _checkpoint_training(cfg, options=None):
    training = dict(checkpoint_retention='latest_then_milestones', save_every=1000,
                    keep_every=1000, keep_from=8000)
    training.update(cfg.get('training', {}))
    training.update((options or {}).get('checkpoint_policy', {}))
    validate_retention(training)
    return training


def _best_validation_record(best, directory, *, saving_step=None, training=None):
    """Historical best scores do not imply that their weights are retained."""
    if best is None:
        return None
    result = dict(best)
    candidate = Path(directory)/f"step_{best['step']:06d}.pt"
    available = candidate.is_file() and not candidate.is_symlink()
    if saving_step is not None:
        available = (best['step'] == saving_step or
                     (available and keep_checkpoint(best['step'], saving_step, training)))
    result.update(checkpoint_available=bool(available), checkpoint=candidate.name if available else None)
    return result


def _complete_pretrain_checkpoint(path, *, stage, step):
    required = {'schema_version', 'stage', 'step', 'memory', 'optimizer', 'rng',
                'provenance', 'options', 'teacher_fit', 'teacher_fitted'}
    if path.is_symlink() or not path.is_file():
        return False
    try:
        document = torch.load(path, map_location='cpu', weights_only=True)
        return (isinstance(document, dict) and required <= document.keys()
                and document['schema_version'] == 4 and document['stage'] == stage
                and document['step'] == step and isinstance(document['memory'], dict)
                and isinstance(document['optimizer'], dict))
    except Exception:
        # An incomplete or unrelated artifact is never a retention target.
        return False


def prune_pretrain_checkpoints(directory, *, stage, latest_step, training):
    """Delete only verified complete checkpoints from this stage after commit."""
    directory = Path(directory)
    newest = directory/f'step_{latest_step:06d}.pt'
    latest_link = directory/'latest.pt'
    if (not latest_link.is_symlink() or latest_link.resolve() != newest.resolve()
            or not _complete_pretrain_checkpoint(newest, stage=stage, step=latest_step)):
        return
    for path in directory.glob('step_*.pt'):
        match = re.fullmatch(r'step_(\d{6,})\.pt', path.name)
        if not match or path.is_symlink() or not path.is_file():
            continue
        step = int(match[1])
        if keep_checkpoint(step, latest_step, training):
            continue
        if (_complete_pretrain_checkpoint(path, stage=stage, step=step)
                and path.is_file() and not path.is_symlink()):
            path.unlink()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_checkpoint_destination(path, *, stage, step, provenance, options):
    """Preserve an unpublished crash artifact before replaying that update."""
    publication = path.with_suffix('.published.json')
    if publication.exists() or publication.is_symlink():
        raise FileExistsError(f'Refusing to replace published checkpoint {path}')
    if not path.exists() and not path.is_symlink():
        return
    latest = path.parent/'latest.pt'
    previous = latest.resolve()
    match = re.fullmatch(r'step_(\d{6,})\.pt', previous.name)
    recoverable = (latest.is_symlink() and previous.parent == path.parent.resolve()
                   and match is not None and int(match[1]) < step
                   and _complete_pretrain_checkpoint(previous, stage=stage, step=int(match[1]))
                   and _complete_pretrain_checkpoint(path, stage=stage, step=step))
    if recoverable:
        for candidate in (previous, path):
            document = torch.load(candidate, map_location='cpu', weights_only=True)
            if document.get('provenance') != provenance or document.get('options') != options:
                recoverable = False
                break
    if not recoverable:
        raise FileExistsError(f'Refusing to replace existing checkpoint {path}')
    evidence = path.parent/f'.{path.stem}.{uuid.uuid4().hex}.unpublished'
    path.rename(evidence)
    _sync_directory(path.parent)


def save_checkpoint(path, *, memory, optimizer, step, stage, cfg, provenance, options,
                    training_rng, query_rng, teacher_fit, best_validation=None, distributed=None,
                    validation=None):
    # All ranks participate before rank zero writes the only checkpoint. Each
    # process has its own CUDA/global RNG state even though data draws are shared.
    rng = _rng_state(memory, training_rng, query_rng)
    rank_rng = distributed.gather(rng) if distributed else [rng]
    if distributed and not distributed.primary:
        return
    path = Path(path)
    _prepare_checkpoint_destination(path, stage=stage, step=step, provenance=provenance, options=options)
    training = _checkpoint_training(cfg, options)
    best_validation = _best_validation_record(best_validation, path.parent,
                                              saving_step=step, training=training)
    document = dict(schema_version=4, stage=stage, config=cfg, memory=memory.state_dict(),
                    teacher_fitted=bool(memory.teacher.fitted), reader_mode=options.get('reader_mode', 'full'),
                    optimizer=optimizer.state_dict(), step=step,
                    provenance=provenance, options=options, teacher_fit=teacher_fit,
                    best_validation=best_validation, rng=rng, rng_by_rank=rank_rng,
                    distributed_world_size=distributed.world_size if distributed else 1)
    if validation is not None:
        if validation.get('step') != step or validation.get('split') != 'validation':
            raise ValueError('Checkpoint validation must describe this exact step on held-out episodes')
        document['validation'] = validation
    # A unique temporary file cannot follow a stale or foreign temporary symlink.
    with tempfile.NamedTemporaryFile(prefix=path.stem+'.', suffix='.tmp',
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(document, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _sync_directory(path.parent)
    latest = path.parent/'latest.pt'
    link = path.parent/f'.latest.{uuid.uuid4().hex}.tmp'
    link.symlink_to(path.name)
    os.replace(link, latest)
    _sync_directory(path.parent)
    # Durable publication evidence prevents replay from ever overwriting a
    # previously published permanent point, even after a manual latest rollback.
    with path.with_suffix('.published.json').open('x') as stream:
        json.dump(dict(stage=stage, step=step, checkpoint=path.name), stream)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)
    # Failed serialization or latest-link replacement never reaches pruning.
    prune_pretrain_checkpoints(path.parent, stage=stage, latest_step=step, training=training)
    _sync_directory(path.parent)
    if best_validation is not None:
        _write_json(path.parent/'best_validation.json',
                    _best_validation_record(best_validation, path.parent))


def restore_checkpoint(path, *, memory, optimizer, stage, provenance, options, training_rng, query_rng,
                       distributed=None):
    document = torch.load(path, map_location='cpu', weights_only=True)
    if document.get('schema_version') != 4 or document.get('stage') != stage:
        raise ValueError('Resume checkpoint stage/schema differs')
    if document.get('provenance') != provenance or document.get('options') != options:
        raise ValueError('Resume input/config/code/options changed; start a new run explicitly')
    world_size = distributed.world_size if distributed else 1
    if document.get('distributed_world_size', 1) != world_size:
        raise ValueError('Exact resume requires the same distributed world size')
    memory.load_state_dict(document['memory'], strict=True)
    if not memory.teacher.fitted.item() or not document.get('teacher_fitted'):
        raise ValueError('Checkpoint teacher is not fitted')
    optimizer.load_state_dict(document['optimizer'])
    rank_rng = document.get('rng_by_rank')
    if rank_rng is not None and len(rank_rng) != world_size:
        raise ValueError('Checkpoint does not contain RNG state for every rank')
    rng = rank_rng[distributed.rank if distributed else 0] if rank_rng is not None else document['rng']
    training_rng.setstate(rng['training'])
    query_rng.set_state(rng['queries'])
    torch.set_rng_state(rng['torch'])
    if rng.get('cuda_local') is not None:
        if next(memory.parameters()).device.type != 'cuda':
            raise ValueError('Exact CUDA resume requested on a CPU-only runtime')
        torch.cuda.set_rng_state(rng['cuda_local'], next(memory.parameters()).device)
    elif rng['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('Exact CUDA resume requested on a CPU-only runtime')
        torch.cuda.set_rng_state_all(rng['cuda'])
    return document


def _synchronize_device(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def _prefix_minimum(memory, value):
    value = memory.codec.slots+1 if value is None else value
    if type(value) is not int or value <= memory.codec.slots:
        raise ValueError('Memory prefix minimum must exceed memory capacity')
    return value


def train_update(memory, optimizer, dataset, stage, *, batch_size, training_rng, query_rng,
                 distributed, seed=0, step=0, max_queries=32, reader_mode='full', min_prefix_decisions=None):
    """One global optimizer update, invariant to partitioning its samples by rank.

    Every rank draws the same *global* list in the same order. Only its assigned
    samples load visual frames or run autograd. A query seed is drawn for every
    global sample, so variable prefix/query lengths cannot change another
    sample's randomness when the GPU count changes.
    """
    if stage != 'memory':
        raise ValueError('Only memory pretraining is supported')
    if batch_size < 1 or batch_size % distributed.world_size:
        raise ValueError('Global --batch-size must be positive and divisible by world size')
    device = distributed.device
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    parameters = [p for p in memory.parameters() if p.requires_grad]
    _synchronize_device(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    rows, identities = [], []
    minimum = _prefix_minimum(memory, min_prefix_decisions)
    for sample in range(batch_size):
        record, endpoint = dataset.sample_prefix(training_rng, min_decisions=minimum)
        query_seed = int(torch.randint(0, 2**63-1, (), generator=query_rng))
        if sample % distributed.world_size != distributed.rank:
            continue
        with loss_rng(seed, stage, step, sample, record['episode_id'], endpoint, device=device):
            loss, metrics = memory_objective(memory, record, endpoint, device=device,
                max_queries=max_queries, generator=torch.Generator().manual_seed(query_seed),
                diagnostics=False, reader_mode=reader_mode, dataset=dataset)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite {stage} loss at step {step}, sample {sample}')
            # sum_gradients uses SUM: global averaging happens exactly once here.
            (loss / batch_size).backward()
        rows.append(metrics)
        identities.append((sample, record['episode_id'], endpoint, query_seed))
    _synchronize_device(device)
    computed = time.perf_counter()
    distributed.sum_gradients(parameters)
    _synchronize_device(device)
    reduced = time.perf_counter()
    gradient_metrics = {}
    for name, module in (('writer', memory.codec.writer), ('reader', memory.decoder)):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        gradient_metrics[name+'_gradient_rms'] = (
            float(torch.sqrt(sum(g.float().square().sum() for g in gradients)/sum(g.numel() for g in gradients)))
            if gradients else 0.)
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f'Nonfinite gradient at step {step}')
    optimizer.step()
    _synchronize_device(device)
    finished = time.perf_counter()
    gathered = distributed.gather(dict(rows=rows, identities=identities, compute_seconds=computed-started,
        gradient_sync_seconds=reduced-computed, optimizer_seconds=finished-reduced,
        update_seconds=finished-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0))
    if not distributed.primary:
        return None
    metrics = _mean_metrics([row for rank in gathered for row in rank['rows']],
                            relative_energy_floor=memory.relative_energy_floor)
    identities = sorted(identity for rank in gathered for identity in rank['identities'])
    durations = {key: max(rank[key] for rank in gathered) for key in
                 ('compute_seconds', 'gradient_sync_seconds', 'optimizer_seconds', 'update_seconds')}
    return dict(gradient_norm=float(gradient_norm), **gradient_metrics, **metrics, **durations,
        peak_allocated_bytes_rank_max=max(rank['peak_allocated_bytes'] for rank in gathered),
        peak_reserved_bytes_rank_max=max(rank['peak_reserved_bytes'] for rank in gathered),
        global_batch_size=batch_size, samples_per_rank=batch_size//distributed.world_size,
        world_size=distributed.world_size, samples_per_second=batch_size/durations['update_seconds'],
        gradient_payload_bytes=sum(p.numel()*p.element_size() for p in parameters) if distributed.world_size > 1 else 0,
        global_sample_sha256=hashlib.sha256(json.dumps(identities).encode()).hexdigest())


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'ablation_memory_capacity/configs/k8.yaml')
    parser.add_argument('--index', type=Path, default=ROOT/'data/history_fixed/index.json')
    parser.add_argument('--stage', choices=['memory'], default='memory')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reader-mode', choices=['full', 'coarse', 'time'], default=None,
                        help='Train an independent residual, coarse-only, or metadata-only control')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Global independent samples per update, divided across torchrun ranks')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--save-every', type=int, default=None,
                        help='Checkpoint interval; defaults to config training.save_every')
    parser.add_argument('--validate-every', type=int, default=None)
    parser.add_argument('--validation-episodes', type=int, default=52)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    supplied = (args.steps, args.batch_size, args.validate_every, args.validation_episodes)
    if any(value is not None and value < 1 for value in supplied) or (args.lr is not None and args.lr <= 0):
        parser.error('Steps, batch size, intervals, validation episodes and learning rate must be positive')
    if args.save_every is not None and args.save_every < 1:
        parser.error('--save-every must be positive')
    distributed = DistributedContext(args.device)
    try:
        run_training(args, distributed)
    except BaseException as error:
        status_path = args.out/'status.json'
        if distributed.primary and status_path.exists():
            status = json.loads(status_path.read_text())
            status.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                          error_type=type(error).__name__)
            _write_json(status_path, status)
        raise
    finally:
        distributed.close()


def run_training(args, distributed):
    if args.stage != 'memory':
        raise ValueError('Only memory pretraining is supported by schema 4')
    cfg = load_config(args.config)
    if cfg.get('schema_version') != 4:
        raise ValueError('Memory pretraining requires schema_version=4')
    if cfg['sft'].get('memory_mode', 'full') != 'full':
        raise ValueError('Memory pretraining requires a full-memory configuration, not the no-memory baseline')
    if (not cfg['memory'].get('residual_enabled', True)
            or cfg['sft'].get('memory_initialization', 'pretrained') == 'random'):
        raise ValueError('This ablation skips memory pretraining')
    stage_settings = cfg['stages']['memory']
    for argument, setting in (('steps', 'steps'), ('batch_size', 'global_batch_size'),
                              ('lr', 'lr'), ('validate_every', 'validate_every')):
        if getattr(args, argument) is None:
            setattr(args, argument, stage_settings[setting])
    if min(args.steps, args.batch_size, args.validate_every, args.validation_episodes) < 1 or args.lr <= 0:
        raise ValueError('Steps, batch size, intervals, validation episodes and learning rate must be positive')
    args.device = str(distributed.device)
    if args.batch_size % distributed.world_size:
        raise ValueError('Global --batch-size must be divisible by world size')
    args.out = args.out.resolve()
    if not args.out.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('--out must be inside this new project runs directory; original runs stay read-only')
    if args.out.exists() and not args.resume:
        raise FileExistsError('Output directory exists; use --resume or a new output directory')
    if args.resume and not (args.out/'latest.pt').exists():
        raise FileNotFoundError('--resume requires an existing latest.pt in --out')
    # No process may create --out before every peer checks its initial state.
    distributed.barrier()
    args.reader_mode = args.reader_mode or cfg['history'].get('reader_mode', 'full')
    # Store the actual independently trained control in architecture/config,
    # not merely in an auxiliary CLI field on an otherwise full-model run.
    cfg['history']['reader_mode'] = args.reader_mode
    training = _checkpoint_training(cfg)
    args.save_every = training['save_every'] if args.save_every is None else args.save_every
    training['save_every'] = args.save_every
    validate_retention(training)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    training_rng = random.Random(args.seed)
    query_rng = torch.Generator().manual_seed(args.seed + 1)
    train = FeatureDataset(args.index, 'train')
    validation = FeatureDataset(args.index, 'validation', document=train.document)
    if train.document['stride'] != cfg['benchmark']['replan_steps']:
        raise ValueError('History index and policy decision stride differ')
    source_hash = file_sha256(cfg['dataset']['manifest'])
    if train.document['source_manifest']['sha256'] != source_hash:
        raise ValueError('Config dataset differs from exported history index')
    input_contract = (validate_input_contract(cfg, index_document=train.document)
                      if distributed.primary else None)
    input_contract = distributed.broadcast(input_contract)
    prefix_dim = train.document['preparation']['teacher']['prefix_dim']
    memory = build_memory(cfg, prefix_dim=prefix_dim).to(args.device)
    teacher_fit = load_teacher_initialization(args.index, memory.teacher, document=train.document)
    parameters = configure_stage(memory, args.stage, args.reader_mode)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(.9, .95), eps=1e-8, weight_decay=0.)
    provenance = provenance_for(args.index, cfg)
    options = dict(stage=args.stage, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                   teacher_artifact=train.document['teacher_initialization'], max_queries=cfg['history']['max_queries'],
                   validation_episodes=args.validation_episodes, validate_every=args.validate_every,
                   save_every=args.save_every, device_type=torch.device(args.device).type, prefix_dim=prefix_dim,
                   reader_mode=args.reader_mode, world_size=distributed.world_size, sampler_version=3,
                   query_sampling=cfg['history']['query_sampling'],
                   checkpoint_policy={key: training[key] for key in
                                      ('checkpoint_retention', 'save_every', 'keep_every', 'keep_from')})
    if 'min_prefix_decisions' in stage_settings:
        options['min_prefix_decisions'] = stage_settings['min_prefix_decisions']
        options['sampler_version'] = 4
    start, best = 0, None
    if args.resume:
        checkpoint = restore_checkpoint(args.out/'latest.pt', memory=memory, optimizer=optimizer,
                         stage=args.stage, provenance=provenance, options=options,
                         training_rng=training_rng, query_rng=query_rng, distributed=distributed)
        start, teacher_fit, best = checkpoint['step'], checkpoint['teacher_fit'], checkpoint.get('best_validation')
        # Preserve evidence of discarded updates instead of silently duplicating steps.
        log = args.out/'updates.jsonl'
        if distributed.primary and log.exists():
            previous = [json.loads(line) for line in log.read_text().splitlines()]
            discarded = [row for row in previous if row['step'] > start]
            if discarded:
                _write_json(args.out/f'discarded_after_step_{start}.json', discarded)
                log.write_text(''.join(json.dumps(row)+'\n' for row in previous if row['step'] <= start))
    # The immutable teacher artifact was prepared before training. All ranks
    # load the same state, then synchronize buffers and parameters explicitly.
    distributed.sync_modules(memory)
    if args.steps <= start:
        raise ValueError(f'--steps must exceed restored step {start}')
    if distributed.primary:
        args.out.mkdir(parents=True, exist_ok=True)
        _write_json(args.out/'run_config.json', dict(config=cfg, options=options, provenance=provenance,
                                                   teacher_fit=teacher_fit, prefix_dim=prefix_dim))
        _write_json(args.out/'input_contract.json', input_contract)
        _write_json(args.out/'status.json', dict(status='running', stage=args.stage, step=start,
                    device=args.device, world_size=distributed.world_size, closed_loop_evaluated=False))
    with TrainingTracker(distributed, args.out, stage=args.stage,
                         config=dict(config=cfg, options=options, provenance=provenance,
                                     teacher_fit=teacher_fit, total_steps=args.steps),
                         resume=args.resume, start_step=start) as tracker:
        memory.train()
        memory.teacher.eval()
        if distributed.primary:
            print(f'Start {args.stage}: step {start+1}/{args.steps}, {sum(p.numel() for p in parameters)} trainable parameters; '
                  f'{distributed.world_size} ranks, global batch {args.batch_size}', flush=True)
        for step in range(start+1, args.steps+1):
            metrics = train_update(memory, optimizer, train, args.stage, batch_size=args.batch_size,
                training_rng=training_rng, query_rng=query_rng, distributed=distributed,
                seed=args.seed, step=step, max_queries=cfg['history']['max_queries'], reader_mode=args.reader_mode,
                min_prefix_decisions=stage_settings.get('min_prefix_decisions'))
            log_record = dict(step=step, stage=args.stage, **metrics) if distributed.primary else None
            validation_report = None
            if step % args.validate_every == 0 or step == args.steps:
                if distributed.primary:
                    result = evaluate(memory, validation, args.stage, episodes=args.validation_episodes,
                                      device=args.device, max_queries=cfg['history']['max_queries'], reader_mode=args.reader_mode,
                                      min_prefix_decisions=stage_settings.get('min_prefix_decisions'))
                    result['step'] = step
                    validation_report = result
                    log_record['validation'] = result['metrics']
                    _write_json(args.out/f'validation_{step:06d}.json', result)
                    score = result['metrics']['history_mse']
                    has_queries = result['metrics'].get('query_elements', 0) > 0
                    if has_queries and score is not None and (best is None or score < best['score']):
                        best = dict(step=step, score=score)
                    best = _best_validation_record(best, args.out)
                    _write_json(args.out/'best_validation.json', best)
                best = distributed.broadcast(best)
            if distributed.primary:
                with (args.out/'updates.jsonl').open('a') as stream:
                    stream.write(json.dumps(log_record, allow_nan=False)+'\n')
                tracker.log(log_record)
                if step % 10 == 0 or step == start+1 or 'validation' in log_record:
                    print(json.dumps(log_record), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                saving_started = time.monotonic()
                save_checkpoint(args.out/f'step_{step:06d}.pt', memory=memory, optimizer=optimizer,
                                step=step, stage=args.stage, cfg=cfg, provenance=provenance, options=options,
                                training_rng=training_rng, query_rng=query_rng, teacher_fit=teacher_fit,
                                best_validation=best, distributed=distributed, validation=validation_report)
                if distributed.primary:
                    save_seconds = time.monotonic()-saving_started
                    with (args.out/'checkpoint_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(step=step, save_seconds=save_seconds))+'\n')
                    tracker.checkpoint(step, save_seconds=save_seconds)
            if distributed.primary:
                _write_json(args.out/'status.json', dict(status='completed' if step == args.steps else 'running',
                           stage=args.stage, step=step, device=args.device, world_size=distributed.world_size,
                           closed_loop_evaluated=False))
        distributed.barrier()


if __name__ == '__main__':
    main()
