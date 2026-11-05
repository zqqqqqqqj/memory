"""Held-out memory acceptance and action-stage input checks."""
import hashlib
import json
import math
from pathlib import Path


class ValidationRejected(RuntimeError):
    training_status = 'validation_rejected'


def memory_validation_gate(report, cfg, step):
    """Require measurable residual information before expensive action training.

    This is a reconstruction check on held-out demonstrations, not evidence of
    improved policy TSR. The checkpoint binds the report to the tested weights.
    """
    minimum = cfg['stages']['memory'].get('min_validation_gain', .01)
    if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum) or not 0 <= minimum < 1):
        raise ValueError('stages.memory.min_validation_gain must be finite in [0, 1)')
    gate = dict(schema_version=1, stage='memory', step=step, passed=False,
                min_validation_gain=minimum, reason='invalid_validation_report')

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    try:
        require(isinstance(report, dict), 'missing checkpoint-bound validation report')
        require(type(step) is int and step > 0 and report.get('step') == step,
                'validation step differs from checkpoint')
        require(report.get('split') == 'validation', 'validation must use the held-out split')
        require(report.get('reader_mode') == 'full' and cfg['history']['objective'] == 'residual',
                'acceptance requires the full residual reconstruction objective')
        identities = report.get('identities')
        episodes = report.get('episodes')
        require(type(episodes) is int and episodes > 0 and isinstance(identities, list)
                and len(identities) == episodes, 'episode count and identities differ')
        for row in identities:
            require(isinstance(row, dict) and isinstance(row.get('episode_id'), str)
                    and bool(row['episode_id']) and type(row.get('task_id')) is int
                    and type(row.get('seed')) is int and type(row.get('current_time')) is int
                    and row['current_time'] >= 0, 'invalid validation episode identity')
        require({row['task_id'] for row in identities} == set(cfg['training']['tasks']),
                'validation does not cover exactly the configured training tasks')
        metrics = report.get('metrics')
        require(isinstance(metrics, dict), 'missing validation metrics')
        names = ('history_queries', 'query_elements', 'delta_sse', 'delta_mse',
                 'target_delta_energy', 'relative_error', 'relative_error_valid',
                 'zero_residual_mse', 'history_sse', 'history_elements', 'history_mse')
        for name in names:
            value = metrics.get(name)
            require(not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value) and value >= 0, f'invalid finite metric: {name}')
        queries, elements = metrics['history_queries'], metrics['query_elements']
        require(queries > 0 and queries == int(queries) and elements > 0
                and elements == queries*cfg['memory']['coarse_dim'],
                'validation has no valid historical query elements')
        energy = metrics['target_delta_energy']
        require(energy > cfg['history']['relative_energy_floor']*elements
                and metrics['relative_error_valid'] == 1, 'validation residual energy is zero or invalid')
        delta_mse = metrics['delta_sse']/elements
        zero_mse = metrics['zero_residual_mse']
        require(zero_mse > 0, 'zero-residual baseline has no positive error')
        for actual, expected in (
                (metrics['delta_mse'], delta_mse),
                (metrics['relative_error'], metrics['delta_sse']/energy),
                (zero_mse, energy/elements),
                (metrics['history_elements'], elements),
                (metrics['history_sse'], metrics['delta_sse']),
                (metrics['history_mse'], delta_mse)):
            require(math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-10),
                    'validation aggregate metrics are inconsistent')
        gain = 1-delta_mse/zero_mse
        passed = gain >= minimum-1e-12
        gate.update(passed=passed, validation_gain=gain, delta_mse=delta_mse,
                    zero_residual_mse=zero_mse, query_elements=elements,
                    target_delta_energy=energy, episodes=episodes,
                    tasks=sorted({row['task_id'] for row in identities}),
                    reason='passed' if passed else 'memory_validation_gain_below_minimum')
    except (KeyError, TypeError, ValueError) as error:
        gate['reason'] = 'invalid_validation_report: '+str(error)
    return gate


def memory_checkpoint_gate(cfg, checkpoint, *, validation_report=None, expected_seed=None):
    """Inspect a published memory checkpoint and its own held-out result on CPU."""
    import torch
    from qrm.evaluation.manifest import sha256
    from qrm.history.config import architecture_signature
    path = Path(checkpoint).resolve(strict=True)
    payload = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('schema_version') != 4 or payload.get('stage') != 'memory':
        raise ValueError('Memory acceptance requires a schema-4 memory checkpoint')
    # Reject incompatible widths/semantics before allocating the action backbone.
    # The saved provenance independently binds the architecture used by memory
    # training; changing only the checkpoint config cannot make it compatible.
    try:
        saved_architecture = architecture_signature(payload['config'])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Memory checkpoint has no valid architecture configuration') from error
    provenance = payload.get('provenance')
    if not isinstance(provenance, dict) or provenance.get('architecture') != saved_architecture:
        raise ValueError('Memory checkpoint architecture differs from its training provenance')
    if saved_architecture != architecture_signature(cfg):
        raise ValueError('Memory pretraining architecture differs from action configuration')
    step = payload.get('step')
    required_step = cfg['stages']['memory'].get('action_checkpoint_step')
    if required_step is not None and step != required_step:
        raise ValueError(f'Action ablation requires the complete memory budget: step {required_step}, got {step}')
    prefix_minimum = cfg['stages']['memory'].get('min_prefix_decisions')
    if prefix_minimum is not None and payload.get('options', {}).get('min_prefix_decisions') != prefix_minimum:
        raise ValueError('Memory checkpoint used a different prefix sampling minimum')
    if expected_seed is not None and payload.get('options', {}).get('seed') != expected_seed:
        raise ValueError('Memory checkpoint seed differs from the action ablation seed')
    publication = json.loads(path.with_suffix('.published.json').read_text())
    if publication != dict(stage='memory', step=step, checkpoint=path.name):
        raise ValueError('Memory checkpoint publication identity differs')
    report = payload.get('validation')
    gate = memory_validation_gate(report, cfg, step)
    if validation_report is not None and validation_report != report:
        gate.update(passed=False, reason='external_validation_report_differs_from_checkpoint')
    if gate['passed']:
        manifest = json.loads(Path(cfg['dataset']['manifest']).read_text())
        records = {row['episode_id']: row for row in manifest['episodes']}
        for identity in report['identities']:
            record = records.get(identity['episode_id'])
            if (record is None or record.get('split') != 'validation'
                    or any(record.get(key) != identity[key] for key in ('task_id', 'seed'))
                    or identity['current_time'] >= record['length']
                    or identity['current_time'] % cfg['benchmark']['replan_steps']):
                gate.update(passed=False, reason='validation_identity_not_in_held_out_manifest')
                break
    gate.update(checkpoint=str(path), checkpoint_sha256=sha256(path))
    if report is not None:
        try:
            encoded = json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        except (TypeError, ValueError):
            gate.update(passed=False, reason='invalid_validation_report: report is not finite JSON')
        else:
            gate['validation_sha256'] = hashlib.sha256(encoded).hexdigest()
    return gate


def validate_action_prerequisites(cfg, memory_checkpoint=None, seed=None):
    """Both action experiments start at the official base; only full needs memory.

    This checks acceptance and architecture before CUDA allocation. The memory
    loader also verifies tensor shapes, the fitted teacher and cache provenance.
    A separately trained action baseline is never an input to either path.
    """
    mode = cfg['sft']['memory_mode']
    if mode == 'none':
        if memory_checkpoint is not None:
            raise ValueError('Independent no-memory baseline must not load a memory checkpoint')
        return dict(mode=mode, backbone_initialization='official_pi05_base', memory_initialization='none')
    if mode != 'full':
        raise ValueError('Action memory_mode must be none or full')
    if cfg['sft'].get('memory_initialization', 'pretrained') == 'random':
        if memory_checkpoint is not None:
            raise ValueError('Random memory initialization must not load a memory checkpoint')
        return dict(mode=mode, backbone_initialization='official_pi05_base',
                    memory_initialization='random_memory_frozen_cache_teacher')
    if memory_checkpoint is None or not Path(memory_checkpoint).is_file():
        raise ValueError('Full memory action training requires a new memory checkpoint')
    gate = memory_checkpoint_gate(cfg, memory_checkpoint,
                                  expected_seed=seed if 'ablation' in cfg else None)
    if not gate['passed']:
        raise ValidationRejected('Memory checkpoint rejected: '+gate['reason'])
    return dict(mode=mode, backbone_initialization='official_pi05_base',
                memory_initialization='pretrained_memory', memory_validation_gate=gate)
