"""Synchronous, held-out closed-loop checks of a published action EMA.

Every training rank launches an independent evaluator on its own physical GPU.
Training model/optimizer/RNG objects never enter the evaluators. Every rank
exchanges subprocess errors before rank zero reads or broadcasts any metrics.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from qrm.config import ROOT
from qrm.evaluation.manifest import sha256
from qrm.joint.evaluation import plan_evaluation


POSE_ENCODING = 'signed_xyzw_to_axis_angle_0_to_2pi'
EVALUATION_SCRIPTS = ('evaluate_history_validation.py', 'run_history_benchmark.py', 'run_benchmark.py')
RANK_ENVIRONMENT = ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                    'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT',
                    'ROLE_NAME', 'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_RESTART_COUNT',
                    'TORCHELASTIC_MAX_RESTARTS', 'TORCHELASTIC_ERROR_FILE', 'TORCHELASTIC_USE_AGENT_STORE')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validation_settings(cfg):
    settings = cfg.get('validation', {})
    if settings.get('enabled') is not True:
        raise ValueError('Synchronous validation must be explicitly enabled')
    tasks = settings.get('tasks', [1, 2, 4, 6])
    count = settings.get('episodes_per_task', 2)
    seed = settings.get('seed_start', 200000)
    timeout = settings.get('timeout_seconds', 1800)
    if (not isinstance(tasks, list) or not tasks or any(type(task) is not int for task in tasks)
            or len(set(tasks)) != len(tasks)):
        raise ValueError('Validation tasks must be distinct integer task IDs')
    if type(count) is not int or count < 1 or type(seed) is not int:
        raise ValueError('Validation episode count and seed must be positive-count integers')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Validation timeout_seconds must be finite and positive')
    # This also rejects test/training seeds and tasks absent from this model.
    plan_evaluation(cfg, split='validation', tasks=tasks, seed_start=seed, episodes_per_task=count)
    return dict(tasks=tasks, episodes_per_task=count, seed_start=seed, timeout_seconds=float(timeout))


def checkpoint_metadata(checkpoint, cfg):
    """Inspect CPU metadata and the durable marker, without allocating a GPU."""
    import torch
    from qrm.joint.checkpoint import complete_checkpoint
    path = Path(checkpoint).resolve(strict=True)
    if not complete_checkpoint(path.parent):
        raise ValueError('Validation requires an atomically published complete checkpoint')
    marker = json.loads((path.parent/'complete.json').read_text())
    if path.name != marker['ema'] or path.stat().st_size != marker['sizes']['ema']:
        raise ValueError('Validation requires the published EMA file')
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if (payload.get('schema_version'), payload.get('stage'), payload.get('history_stage'),
            payload.get('weights')) != (4, 'history_sft', 'action', 'complete_ema'):
        raise ValueError('Validation requires a complete schema-4 action EMA')
    if payload.get('step') != marker['step'] or payload.get('config') != cfg:
        raise ValueError('EMA step/config differs from the training validation request')
    training_sources = payload.get('source_signature')
    if not isinstance(training_sources, dict) or not training_sources:
        raise ValueError('Checkpoint lacks its training source fingerprint')
    for name, digest in training_sources.items():
        source = (ROOT/name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not source.is_file() or sha256(source) != digest:
            raise ValueError(f'Checkpoint source differs from the validation implementation: {name}')
    return dict(checkpoint=str(path), checkpoint_step=payload['step'], checkpoint_sha256=sha256(path),
                checkpoint_size=path.stat().st_size, weights='complete_ema',
                method=payload['method'], memory_mode=cfg['sft'].get('memory_mode', 'full'),
                policy_seed=payload['seed'], pose_encoding=POSE_ENCODING,
                norm_stats_sha256=payload['norm_stats_sha256'],
                source_signature=training_sources,
                evaluation_sources={f'scripts/{name}': sha256(ROOT/'scripts'/name)
                                    for name in EVALUATION_SCRIPTS})


def build_validation_plan(cfg, metadata, world_size):
    settings = validation_settings(cfg)
    if type(world_size) is not int or world_size < 1:
        raise ValueError('Validation world size must be a positive integer')
    common = dict(split='validation', tasks=settings['tasks'], seed_start=settings['seed_start'],
                  episodes_per_task=settings['episodes_per_task'])
    full = plan_evaluation(cfg, **common)
    shards = []
    for rank in range(world_size):
        shard = plan_evaluation(cfg, **common, shard_index=rank, shard_count=world_size)
        shard.update(checkpoint=metadata['checkpoint'], checkpoint_sha256=metadata['checkpoint_sha256'],
                     checkpoint_step=metadata['checkpoint_step'], weights='complete_ema')
        shards.append(shard)
    # Normalize tuples once, so saved and in-memory plans compare exactly.
    return json.loads(json.dumps(dict(schema_version=1, **metadata, split='validation',
        purpose='training_checkpoint_validation', tasks=full['tasks'], seeds=full['seeds'],
        total_episodes=full['total_episodes'], world_size=world_size,
        timeout_seconds=settings['timeout_seconds'], shards=shards)))


def validation_environment(ctx, environment=None):
    """Map training local rank to its physical card, then expose only cuda:0."""
    environment = dict(os.environ if environment is None else environment)
    local_rank = int(getattr(ctx, 'local_rank', ctx.rank))
    visible = environment.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        physical = str(local_rank)
    else:
        cards = visible.split(',')
        if local_rank < 0 or local_rank >= len(cards):
            raise ValueError('Training local rank is outside CUDA_VISIBLE_DEVICES')
        physical = cards[local_rank].strip()
    if not physical.isdecimal():
        raise ValueError('Validation requires numeric physical GPUs for matching EGL selection')
    for name in RANK_ENVIRONMENT:
        environment.pop(name, None)
    environment.update(CUDA_VISIBLE_DEVICES=physical, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       MUJOCO_EGL_DEVICE_ID=physical, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                       JAX_PLATFORMS='cpu', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1',
                       OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'), environment.get('PYTHONPATH', '')])
    return environment


def stop_process_group(process):
    """The evaluator owns a new session; its simulator belongs to the same group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # The worker may exit before its simulator. Clear the whole owned group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def run_worker_process(command, *, environment, log_path, timeout_seconds):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout_seconds)
            if code:
                raise RuntimeError(f'Validation worker exited with code {code}; inspect {log_path}')
        except BaseException as error:
            stop_process_group(process)
            if isinstance(error, subprocess.TimeoutExpired):
                raise TimeoutError(f'Validation exceeded {timeout_seconds:g}s; inspect {log_path}') from error
            raise


def validate_shard_results(rows, shard):
    if not isinstance(rows, list) or len(rows) != len(shard['jobs']):
        raise ValueError('Validation shard is missing or has extra episode results')
    seen = set()
    for row, job, trial in zip(rows, shard['jobs'], shard['trial_indices'], strict=True):
        if not isinstance(row, dict):
            raise ValueError('Validation episode must be an object')
        identity = (row.get('task_id'), row.get('seed'), row.get('trial_index'))
        if (any(type(value) is not int for value in identity) or identity in seen
                or identity != (job[0], job[1], trial)):
            raise ValueError('Validation episode identity is duplicate, out of order or outside the plan')
        seen.add(identity)
        if row.get('status') != 'completed' or row.get('errors'):
            raise ValueError('Technical rollout errors are not valid policy failures')
        for name in ('TSR', 'CSR'):
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid episode {name}')
        if row['TSR'] not in (0, 1):
            raise ValueError('Episode TSR must be binary')
    return rows


def collect_validation_report(out, plan):
    out = Path(out)
    rows = []
    for rank, expected in enumerate(plan['shards']):
        directory = out/f'rank_{rank}'
        if json.loads((directory/'evaluation_plan.json').read_text()) != expected:
            raise ValueError(f'Validation rank {rank} plan/provenance mismatch')
        receipt = json.loads((directory/'worker_complete.json').read_text())
        if receipt != dict(rank=rank, checkpoint_sha256=plan['checkpoint_sha256'],
                           episodes=len(expected['jobs']), status='completed'):
            raise ValueError(f'Validation rank {rank} completion receipt mismatch')
        rows.extend(validate_shard_results(json.loads((directory/'episodes.json').read_text()), expected))
    if len(rows) != plan['total_episodes']:
        raise ValueError('Validation episode total differs from its plan')
    per_task = {}
    for task in plan['tasks']:
        episodes = [row for row in rows if row['task_id'] == task]
        if len(episodes) != len(plan['seeds']):
            raise ValueError('Validation task coverage is incomplete')
        per_task[str(task)] = dict(episodes=len(episodes), **{
            metric: sum(row[metric] for row in episodes)/len(episodes) for metric in ('TSR', 'CSR')})
    metadata = {key: plan[key] for key in ('checkpoint', 'checkpoint_sha256', 'checkpoint_size',
        'checkpoint_step', 'weights', 'method', 'memory_mode', 'policy_seed', 'pose_encoding',
        'norm_stats_sha256', 'source_signature', 'evaluation_sources', 'tasks', 'seeds', 'total_episodes')}
    return dict(schema_version=1, status='completed', split='validation', **metadata, per_task=per_task,
                macro_TSR=sum(row['TSR'] for row in per_task.values())/len(per_task),
                macro_CSR=sum(row['CSR'] for row in per_task.values())/len(per_task))


def evaluate_history_checkpoint(checkpoint, cfg, ctx, out):
    """All ranks call this after checkpoint publication; all return or raise."""
    out = Path(out).resolve()
    prepared = None
    if ctx.primary:
        try:
            metadata = checkpoint_metadata(checkpoint, cfg)
            plan = build_validation_plan(cfg, metadata, ctx.world_size)
            out.mkdir(parents=True, exist_ok=True)
            saved = out/'validation_plan.json'
            if saved.exists() and json.loads(saved.read_text()) != plan:
                raise ValueError('Existing validation output belongs to another plan/checkpoint')
            atomic_json(saved, plan)
            atomic_json(out/'config.json', cfg)
            atomic_json(out/'status.json', dict(status='running', checkpoint_step=plan['checkpoint_step']))
            prepared = dict(plan=plan, error=None)
        except Exception as error:
            prepared = dict(error=f'{type(error).__name__}: {error}')
    prepared = ctx.broadcast(prepared)
    if prepared['error']:
        raise RuntimeError('Validation preparation failed: '+prepared['error'])
    plan = prepared['plan']
    local_error = None
    try:
        environment = validation_environment(ctx)
        command = [sys.executable, '-u', str(ROOT/'scripts/evaluate_history_validation.py'),
                   '--out', str(out), '--rank', str(ctx.rank)]
        run_worker_process(command, environment=environment, log_path=out/f'rank_{ctx.rank}.console.log',
                           timeout_seconds=plan['timeout_seconds'])
    except Exception as error:
        local_error = f'rank {ctx.rank}: {type(error).__name__}: {error}'
    errors = ctx.gather(local_error)
    outcome = None
    if ctx.primary:
        try:
            if any(errors):
                raise RuntimeError('; '.join(error for error in errors if error))
            report = collect_validation_report(out, plan)
            atomic_json(out/'metrics.json', report)
            atomic_json(out/'status.json', dict(status='completed', checkpoint_step=plan['checkpoint_step']))
            outcome = dict(report=report, error=None)
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            try:
                atomic_json(out/'status.json', dict(status='failed', error=message,
                                                   checkpoint_step=plan['checkpoint_step']))
            except OSError:
                pass
            outcome = dict(error=message)
    outcome = ctx.broadcast(outcome)
    if outcome['error']:
        raise RuntimeError('Closed-loop validation failed: '+outcome['error'])
    return outcome['report']


def validation_gate(report, cfg, step):
    """A configured control-quality gate; thresholds never use test seeds."""
    settings = cfg.get('validation', {})
    threshold_step = settings.get('require_positive_tsr_by_step', 3000)
    minimum = settings.get('min_macro_tsr', 0.)
    if type(threshold_step) is not int or threshold_step < 0:
        raise ValueError('Validation gate step must be a nonnegative integer')
    if isinstance(minimum, bool) or not isinstance(minimum, (int, float)) or not math.isfinite(minimum) or not 0 <= minimum < 1:
        raise ValueError('Validation gate min_macro_tsr must be finite in [0,1)')
    score = report.get('macro_TSR')
    if (report.get('status') != 'completed' or report.get('split') != 'validation'
            or report.get('checkpoint_step') != step or isinstance(score, bool)
            or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
        raise ValueError('Validation gate requires the completed current validation checkpoint')
    required = settings.get('enabled') is True and step >= threshold_step
    passed = not required or score > minimum
    return dict(required=required, passed=passed, step=step, min_macro_tsr=minimum, macro_TSR=score,
                reason='passed' if passed else 'validation_macro_TSR_did_not_exceed_configured_minimum')
