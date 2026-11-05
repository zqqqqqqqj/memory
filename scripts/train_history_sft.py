"""Schema-4 action SFT: fixed-feature history and shared residual supervision."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from qrm.config import ROOT, load_config
from qrm.data import load_manifest, validate_seed_partition
from qrm.distributed import DistributedContext, loss_rng
from qrm.evaluation.manifest import sha256, write_manifest
from qrm.history.config import resolve_stage
from qrm.history.compact_cache import read_feature_index
from qrm.history.data import IndexedFeatureCache, load_teacher_initialization
from qrm.history.acceptance import validate_action_prerequisites, ValidationRejected
from qrm.history.validation import evaluate_history_checkpoint, validation_gate
from qrm.history.training_validation import publish_training_validation, resume_training_validation
from qrm.history.input_contract import validate_input_contract
from qrm.history.joint import (HistoryJointModel, configure_backbone, load_memory_initialization,
                               optimizer_groups)
from qrm.history.performance import accumulate_metrics, gradient_rms, reduce_metric_sums
from qrm.history.tracking import TrainingTracker
from qrm.joint.checkpoint import (ShardEMA, save_joint_checkpoint, latest_complete_checkpoint,
                                  restore_joint_checkpoint)
from qrm.joint.config import accumulation_steps
from qrm.history.input_pipeline import prepare_cpu_history, materialize_history, prefetch_histories
from qrm.joint.logging import write_training_status
from qrm.joint.parallel import wrap_sft, accumulation_context, communication_metadata
from qrm.joint.sampling import select_decision
from qrm.joint.model import frozen_visual
from qrm.normalization import validate_norm_stats
from qrm.policies.pi05_bridge import Pi05Bridge

TIMING_SEMANTICS = ('Host wall-clock phases; CUDA is asynchronous and pending GPU work can be charged '
                    'to optimizer/reduce. step_seconds includes the complete update and excludes checkpoint saving.')


def read_index(path, cfg, base_hash, norm_hash, *, document=None):
    index = document if document is not None else read_feature_index(path)
    provenance = index['visual_provenance']
    expected = dict(base_sha256=base_hash, norm_stats_sha256=norm_hash,
                    source_manifest_sha256=sha256(cfg['dataset']['manifest']),
                    preprocess_sha256=sha256(ROOT/'src/qrm/policies/pi05_bridge.py'),
                    encoder_sha256=hashlib.sha256(inspect.getsource(frozen_visual).encode()).hexdigest(),
                    openpi_commit=cfg['model']['openpi_commit'])
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f'Frozen history cache provenance mismatch: {key}')
    if index['stride'] != cfg['benchmark']['replan_steps']:
        raise ValueError('Cache observation frequency differs from policy protocol')
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'ablation_memory_capacity/configs/k8.yaml'))
    parser.add_argument('--stage', choices=('action',), default='action')
    parser.add_argument('--history-index', type=Path, required=True)
    parser.add_argument('--memory-checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--tasks', nargs='+', type=int)
    parser.add_argument('--steps', type=int, help='Explicit override of experimental stage budget')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-pipeline', choices=('cached_prefetch', 'synchronous'),
                        help='Ordered CPU prefetch or the equivalent sequential reference')
    parser.add_argument('--prefetch-depth', type=int, help='Bounded ready-prefix queue per rank; default 1')
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        parser.error('--steps must be positive')
    if args.prefetch_depth is not None and args.prefetch_depth < 1:
        parser.error('--prefetch-depth must be positive')
    cfg = resolve_stage(load_config(args.config), args.stage, args.tasks)
    cfg['sft']['input_pipeline'] = args.input_pipeline or cfg['sft'].get('input_pipeline', 'cached_prefetch')
    cfg['sft']['prefetch_depth'] = args.prefetch_depth or cfg['sft'].get('prefetch_depth', 1)
    if args.steps is not None:
        cfg['sft']['steps'] = args.steps
        cfg['sft']['warmup_steps'] = min(cfg['sft']['warmup_steps'], args.steps)
    # Check the CPU metadata and exact preprocessing sources before creating a
    # CUDA context or loading the large policy. Every rank checks the same files.
    index_document = read_feature_index(args.history_index, check_files=False)
    input_contract = validate_input_contract(cfg, index_document)
    prerequisites = validate_action_prerequisites(cfg, args.memory_checkpoint, args.seed)
    memory_hash = sha256(args.memory_checkpoint) if args.memory_checkpoint else None
    context = DistributedContext(args.device)
    if context.device.type != 'cuda':
        raise ValueError('Full pi05 training requires CUDA; run CPU tests for graph validation')
    out = args.out.resolve()
    # Standalone stages must never reuse a parent project output directory.
    if out == (ROOT/'runs').resolve() or not out.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Write new stage outputs beneath this project runs/ directory')
    if context.broadcast(out.exists() if context.primary else None) and not args.resume:
        raise FileExistsError('Use a fresh output directory or --resume')
    accum = accumulation_steps(cfg, context.world_size)
    communication = communication_metadata(cfg['sft'], context.world_size, accum)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    norms = validate_norm_stats(cfg['dataset']['norm_stats'], cfg['dataset']['manifest'])
    base = cfg['model']['base_checkpoint_pytorch']
    base_hash = context.broadcast(sha256(Path(base)/'model.safetensors') if context.primary else None)
    # Only rank zero checks all compact episode files. Every peer has already
    # checked the immutable index and current preprocessing contract.
    validation_result = None
    if context.primary:
        try:
            validation_result = dict(document=read_feature_index(args.history_index), error=None)
        except Exception as error:
            validation_result = dict(document=None, error=f'{type(error).__name__}: {error}')
    validation_result = context.broadcast(validation_result)
    if validation_result['error'] is not None:
        raise ValueError('Fixed-feature index preflight failed: '+validation_result['error'])
    verified_index = validation_result['document']
    index = read_index(args.history_index, cfg, base_hash, norms['norm_stats_sha256'], document=verified_index)
    records = load_manifest(cfg['dataset']['manifest'])
    validate_seed_partition(records, cfg['benchmark'], 'train')
    records = [r for r in records if r['task_id'] in cfg['training']['tasks']]
    covered = {r['episode_id'] for r in index['episodes'] if r['split'] == 'train'}
    if {r['episode_id'] for r in records} - covered:
        raise ValueError('History index must contain every selected training episode; export complete prefixes first')
    bridge = Pi05Bridge(base, device=str(context.device), norm_stats=norms['norm_stats'], action_horizon=20)
    model = HistoryJointModel(bridge, cfg).to(context.device)
    if (cfg['sft']['memory_mode'] == 'full'
            and cfg['sft'].get('memory_initialization', 'pretrained') == 'pretrained'):
        memory_payload = load_memory_initialization(model, args.memory_checkpoint)
        if memory_payload.get('provenance', {}).get('index_sha256') != sha256(args.history_index):
            raise ValueError('Memory pretraining used a different history index')
        if memory_payload.get('reader_mode', 'full') != 'full':
            raise ValueError('Action training requires the full residual memory, not a diagnostic control')
        del memory_payload
    else:
        # Reuse only the fixed cache projection/normalization. The learned
        # writer and reader remain freshly initialized in no-pretraining runs.
        load_teacher_initialization(args.history_index, model.memory.teacher, document=index)
    cache = IndexedFeatureCache(args.history_index, model.memory.teacher, document=index, check_files=False)
    # The backbone starts at the official base and the full-memory branch is
    # active from the first action update. Configure FP32 master parameters
    # before any same-run EMA/optimizer restoration through --resume.
    pattern_counts = configure_backbone(model.backbone, cfg['sft']['backbone_train_patterns'])
    initialization = 'official_pi05_base'
    model.configure_stage(args.stage)
    extras = sum(p.numel() for p in model.small_parameters())
    if extras > cfg['memory']['max_extra_parameters']:
        raise ValueError(f'Memory/interface parameter budget exceeded: {extras}')
    source_files = [*sorted((ROOT/'src/qrm').rglob('*.py')), Path(__file__).resolve()]
    method = ('action_no_memory' if model.memory_mode == 'none' else
              'history_residual' if cfg['memory'].get('residual_enabled', True) else 'history_coarse_only')
    metadata = dict(config=cfg, method=method, seed=args.seed, backbone=base,
        backbone_sha256=base_hash, **norms, data_manifest_sha256=sha256(cfg['dataset']['manifest']),
        plan_sha256=sha256(ROOT/cfg['plan']), extra_parameters=extras,
        actual_world_size=context.world_size, actual_accumulation_steps=accum,
        history_stage=args.stage, history_index_sha256=sha256(args.history_index),
        memory_checkpoint_sha256=memory_hash,
        initial_checkpoint_sha256=None,
        initialization=initialization, automatic_validation=True, backbone_train_patterns=pattern_counts,
        action_prerequisites=prerequisites,
        input_contract=input_contract,
        communication=communication, input_pipeline=cfg['sft']['input_pipeline'],
        prefetch_depth=cfg['sft']['prefetch_depth'], vision_batch_size=1,
        timing_semantics=TIMING_SEMANTICS,
        source_signature={str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        loss_definition=('OpenPI padded action mean' +
            (f" + {cfg['history']['loss_weight']} * block-local fixed-feature {cfg['history']['objective']} MSE"
             if cfg['history'].get('joint_supervision', True) and model.memory_mode == 'full' else '') +
            '; actual 7d/padding reported separately'))
    if context.primary and not args.resume:
        out.mkdir(parents=True)
        (out/'train_config.json').write_text(json.dumps(metadata, indent=2)+'\n')
        (out/'input_contract.json').write_text(json.dumps(input_contract, indent=2)+'\n')
        (out/'parameter_audit.json').write_text(json.dumps([
            dict(name=n, parameters=p.numel(), trainable=p.requires_grad, dtype=str(p.dtype))
            for n, p in model.named_parameters()], indent=2)+'\n')
        write_manifest(cfg, out/'versions.json', config_path=args.config, role='history_sft', extra=metadata)
    context.barrier()
    if context.world_size > 1:
        model = wrap_sft(model, cfg['sft'], context.device)
    original = model.module if isinstance(model, FSDP) else model
    groups = optimizer_groups(model, cfg)
    optimizer = torch.optim.AdamW(groups, betas=tuple(cfg['sft']['betas']), eps=cfg['sft']['eps'],
                                  weight_decay=cfg['sft']['weight_decay'])
    start = 0
    if args.resume:
        checkpoint = Path(context.broadcast(str(latest_complete_checkpoint(out)) if context.primary else None))
        start, ema = restore_joint_checkpoint(model, optimizer, context, checkpoint, metadata)
        if context.primary:
            # Preserve discarded updates separately after an interrupted save interval.
            log = out/'updates.jsonl'
            if log.exists():
                lines = log.read_text().splitlines()
                kept, discarded = [], []
                for line in lines:
                    try:
                        destination = kept if json.loads(line)['step'] <= start else discarded
                    except (ValueError, KeyError):
                        destination = discarded
                    destination.append(line)
                if discarded:
                    (out/f'updates_before_resume_{time.time_ns()}.jsonl').write_text('\n'.join(discarded)+'\n')
                    log.write_text('\n'.join(kept)+'\n')
    else:
        ema = ShardEMA(model, cfg['sft']['ema_decay'])
    if cfg['sft']['steps'] < start:
        raise ValueError(f'Action training budget must not precede restored step {start}')
    # The producer runs CPU-only deterministic transforms. Resume starts at the
    # first unconsumed global index; abandoned lookahead never skips examples.
    inputs = (prefetch_histories(bridge, records, args.seed,
              start*cfg['sft']['global_batch_size']+context.rank,
              cfg['sft']['steps']*cfg['sft']['global_batch_size'], context.world_size,
              cfg['benchmark']['replan_steps'], cache, capacity=cfg['sft']['prefetch_depth'])
              if cfg['sft']['input_pipeline'] == 'cached_prefetch' else None)
    if context.primary:
        print('Communication: '+json.dumps(communication), flush=True)
        print(f"Input pipeline: {cfg['sft']['input_pipeline']}; queue depth={cfg['sft']['prefetch_depth']}; "
              'compact FP32 history; current full visual token only; complete small-memory history gradients', flush=True)
    last = start
    quality_rejected = False
    try:
        with TrainingTracker(context, out, stage=args.stage,
                             config=dict(config=cfg, seed=args.seed, world_size=context.world_size,
                                         global_batch_size=cfg['sft']['global_batch_size'],
                                         accumulation_steps=accum, initialization=initialization,
                                         communication=communication, backbone_train_patterns=pattern_counts,
                                         extra_parameters=extras),
                             resume=args.resume, start_step=start) as tracker:
            if args.resume:
                # Saving precedes evaluation. A complete checkpoint alone must
                # not let a restart skip a pending or rejected quality gate,
                # including the final step where no optimizer work remains.
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                resumed = resume_training_validation(out, checkpoint/'ema.pt', cfg, context, start)
                if resumed is not None:
                    report, gate = resumed['report'], resumed['gate']
                    quality_rejected = not gate['passed']
                    publish_training_validation(out, resumed['directory'], report, gate,
                                                context, tracker, start)
                    if context.primary:
                        print(json.dumps(dict(resume_validation_step=start, mode=resumed['mode'],
                                              previous_attempt=resumed.get('previous_attempt'))), flush=True)
                    if quality_rejected:
                        raise ValidationRejected(f"Restored step {start}: validation macro TSR "
                            f"{report['macro_TSR']:.4f} must exceed {gate['min_macro_tsr']}; saved EMA retained")
            for step in range(start+1, cfg['sft']['steps']+1):
                beginning = time.monotonic()
                torch.cuda.reset_peak_memory_stats(context.device)
                optimizer.zero_grad(set_to_none=True)
                totals = {}
                timing = dict(input_wait_seconds=0., host_visual_prepare_seconds=0., host_forward_backward_seconds=0.,
                              cpu_prepare_seconds=0., cache_hits=0, cache_misses=0, prepared_image_decisions=0,
                              fixed_feature_frames=0, fixed_feature_bytes=0)
                for micro in range(accum):
                    sample_index = (step-1)*cfg['sft']['global_batch_size']+micro*context.world_size+context.rank
                    phase = time.monotonic()
                    if inputs is not None:
                        actual_index, _, cpu = next(inputs)
                        if actual_index != sample_index:
                            raise RuntimeError(f'CPU prefetch sample order changed: {actual_index} != {sample_index}')
                    else:
                        record, timestamp = select_decision(records, args.seed, sample_index, cfg['benchmark']['replan_steps'])
                        cpu = prepare_cpu_history(bridge, record, timestamp, cfg['benchmark']['replan_steps'], cache)
                    timing['input_wait_seconds'] += time.monotonic()-phase
                    phase = time.monotonic()
                    sample = materialize_history(bridge, cpu, cache)
                    timing['host_visual_prepare_seconds'] += time.monotonic()-phase
                    for name in ('cpu_prepare_seconds', 'cache_hits', 'cache_misses', 'prepared_image_decisions',
                                 'fixed_feature_frames', 'fixed_feature_bytes'):
                        timing[name] += cpu.metrics[name]
                    phase = time.monotonic()
                    with accumulation_context(model, micro, accum, cfg['sft']['gradient_sync']):
                        with loss_rng(args.seed, sample_index, device=context.device):
                            loss, metrics = model(sample)
                            (loss/accum).backward()
                    accumulate_metrics(totals, dict(loss=loss.detach(), **metrics), context.device)
                    timing['host_forward_backward_seconds'] += time.monotonic()-phase
                    del sample, cpu, loss
                phase = time.monotonic()
                gradients = gradient_rms(context, {'writer': original.memory.codec.writer,
                                                   'reader': original.memory.decoder})
                gradient = (model.clip_grad_norm_(cfg['sft']['max_grad_norm']) if isinstance(model, FSDP)
                            else torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['sft']['max_grad_norm'], error_if_nonfinite=True))
                if not torch.isfinite(gradient):
                    raise FloatingPointError('Nonfinite accumulated gradient')
                for group in optimizer.param_groups:
                    group['lr'] = group['peak_lr']*min(step/max(cfg['sft']['warmup_steps'], 1), 1.)
                optimizer.step()
                ema.update(model)
                totals = reduce_metric_sums(context, totals, cfg['sft']['global_batch_size'],
                                            history_energy_floor=cfg['history']['relative_energy_floor'])
                timing['host_optimizer_and_reduce_seconds'] = time.monotonic()-phase
                timing['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(context.device)
                timing['peak_reserved_bytes'] = torch.cuda.max_memory_reserved(context.device)
                timing['step_seconds'] = time.monotonic()-beginning
                rank_timings = context.gather(timing)
                last = step
                if context.primary:
                    row = dict(step=step, stage=args.stage, step_seconds=time.monotonic()-beginning,
                               gradient_norm=float(gradient), writer_gradient_rms=gradients['writer'],
                               reader_gradient_rms=gradients['reader'], lr=[group['lr'] for group in optimizer.param_groups],
                               communication=communication, input_pipeline=cfg['sft']['input_pipeline'],
                               timing_semantics=TIMING_SEMANTICS,
                               timing_by_rank=rank_timings, **totals)
                    for name in ('input_wait_seconds', 'host_visual_prepare_seconds', 'host_forward_backward_seconds',
                                 'host_optimizer_and_reduce_seconds', 'cpu_prepare_seconds'):
                        row[name+'_rank_mean'] = sum(item[name] for item in rank_timings)/context.world_size
                    row['peak_allocated_bytes_rank_max'] = max(item['peak_allocated_bytes'] for item in rank_timings)
                    row['peak_reserved_bytes_rank_max'] = max(item['peak_reserved_bytes'] for item in rank_timings)
                    hits, misses = (sum(item[key] for item in rank_timings) for key in ('cache_hits', 'cache_misses'))
                    row['visual_cache_hit_fraction'] = hits/max(hits+misses, 1)
                    row['samples_per_second'] = cfg['sft']['global_batch_size']/row['step_seconds']
                    with (out/'updates.jsonl').open('a') as stream:
                        stream.write(json.dumps(row)+'\n')
                    write_training_status(out, 'running', step)
                    tracker.log(row)
                    print(json.dumps(row), flush=True)
                if step % cfg['training']['save_every'] == 0 or step == cfg['sft']['steps']:
                    saving_started = time.monotonic()
                    save_joint_checkpoint(model, optimizer, ema, context, out, step, metadata)
                    if context.primary:
                        with (out/'checkpoint_events.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(step=step, save_seconds=time.monotonic()-saving_started))+'\n')
                        tracker.checkpoint(step, save_seconds=time.monotonic()-saving_started)
                if step % cfg['sft']['eval_every'] == 0 or step == cfg['sft']['steps']:
                    # All ranks wait while isolated evaluators use the durable
                    # EMA. Never overwrite live optimizer/model/RNG state.
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    validation_out = out/'validation'/f'step_{step:06d}'
                    report = evaluate_history_checkpoint(out/'checkpoints'/f'step_{step:06d}'/'ema.pt',
                                                         cfg, context, validation_out)
                    gate = validation_gate(report, cfg, step)
                    quality_rejected = not gate['passed']
                    publish_training_validation(out, validation_out, report, gate, context, tracker, step)
                    if quality_rejected:
                        raise ValidationRejected(f"Step {step}: validation macro TSR {report['macro_TSR']:.4f} "
                                                 f"must exceed {gate['min_macro_tsr']}; saved EMA retained")
            if context.primary:
                write_training_status(out, 'completed', last)
    except BaseException as error:
        if context.primary:
            status = 'interrupted' if isinstance(error, KeyboardInterrupt) else getattr(error, 'training_status', 'failed')
            write_training_status(out, status, last, error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        try:
            if inputs is not None:
                inputs.close()
        finally:
            try:
                # A known all-rank quality decision is different from an
                # arbitrary runtime failure: peers must let rank zero finish
                # Tracking.__exit__ and persist the rejected status before
                # any nonzero exit makes torchrun terminate its siblings.
                if quality_rejected:
                    context.barrier()
            finally:
                context.close()


if __name__ == '__main__':
    main()
