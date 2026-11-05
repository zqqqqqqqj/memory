"""Schema 4: fixed features and two-stage residual-supervised memory."""
import copy
import math
import struct

from qrm.checkpoint_retention import validate_retention


WRITER_VERSION = 'bounded_count_convex_v2'
MEMORY_INTERFACE_VERSION = 'separate_capped_v2'


def validate_writer_bound(value, storage_dtype='float32'):
    """Require an exact finite endpoint, without importing a model or torch.

    If B is a representable endpoint, rounding an in-range FP32 merge to the
    configured persistent dtype cannot round past B. In particular, 0.1 must
    not silently become a larger BF16 bound than the configured number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('writer_bound must be a positive finite number')
    if storage_dtype not in {'float32', 'bfloat16'}:
        raise ValueError('Unsupported writer bound storage dtype')
    try:
        packed = struct.pack('>f', value)
    except (OverflowError, struct.error) as error:
        raise ValueError('writer_bound must be exactly representable in its storage dtype') from error
    exact_fp32 = struct.unpack('>f', packed)[0]
    bits = int.from_bytes(packed, 'big')
    if exact_fp32 != value or (storage_dtype == 'bfloat16' and bits & 0xFFFF):
        raise ValueError('writer_bound must be exactly representable in its storage dtype')
    return float(value)


def validate_action_validation(cfg):
    """Require a small, explicit closed-loop check tied to saved action weights."""
    validation = cfg.get('validation')
    if not isinstance(validation, dict) or validation.get('enabled') is not True:
        raise ValueError('Action training requires validation.enabled=true')
    every, save = cfg['sft'].get('eval_every'), cfg['training'].get('save_every')
    if (type(every) is not int or every < 1 or type(save) is not int or save < 1
            or every % save):
        raise ValueError('sft.eval_every must be a positive multiple of training.save_every')
    tasks = validation.get('tasks')
    if (not isinstance(tasks, (list, tuple)) or not tasks or any(type(task) is not int for task in tasks)
            or len(set(tasks)) != len(tasks) or not set(tasks) <= set(cfg['training']['tasks'])):
        raise ValueError('Validation tasks must be distinct and covered by the training tasks')
    count, start = validation.get('episodes_per_task'), validation.get('seed_start')
    low, high = cfg['benchmark']['validation_seeds']
    if (type(count) is not int or count < 1 or type(start) is not int
            or start < low or start + count - 1 > high):
        raise ValueError('Validation episodes must use consecutive registered validation seeds')
    timeout = validation.get('timeout_seconds')
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Validation timeout_seconds must be positive and finite')
    threshold_step = validation.get('require_positive_tsr_by_step')
    if type(threshold_step) is not int or threshold_step < 1:
        raise ValueError('Validation require_positive_tsr_by_step must be a positive step')
    minimum = validation.get('min_macro_tsr')
    if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum) or not 0 <= minimum < 1):
        raise ValueError('Validation min_macro_tsr must lie in [0, 1)')


def validate_config(cfg):
    if cfg.get('schema_version') != 4:
        raise ValueError('History supervision requires schema_version=4')
    if 'ppo' in cfg:
        raise ValueError('Schema 4 has no PPO stage or Gaussian latent configuration')
    model, memory, history, sft = (cfg[k] for k in ('model', 'memory', 'history', 'sft'))
    if 'ablation' in cfg:
        protocol = cfg['ablation']
        if (not isinstance(protocol, dict) or protocol.get('protocol') != 'table2_v1'
                or protocol.get('parameter_initialization') != 'named_linear_v1'):
            raise ValueError('Unknown table-2 ablation protocol or parameter initialization')
    if model.get('lora', False):
        raise ValueError('This configuration uses ordinary selected-layer training, not LoRA')
    mode = sft.get('memory_mode', 'full')
    if mode not in {'none', 'full'}:
        raise ValueError('sft.memory_mode must be none or full')
    residual_enabled = memory.get('residual_enabled', True)
    if type(residual_enabled) is not bool:
        raise ValueError('memory.residual_enabled must be a boolean')
    initialization = sft.get('memory_initialization', 'pretrained')
    if initialization not in {'pretrained', 'random'}:
        raise ValueError('sft.memory_initialization must be pretrained or random')
    joint_supervision = history.get('joint_supervision', True)
    if type(joint_supervision) is not bool:
        raise ValueError('history.joint_supervision must be a boolean')
    if not joint_supervision and history['loss_weight'] != 0:
        raise ValueError('Disabled joint supervision requires history.loss_weight=0')
    if mode == 'full' and not residual_enabled and (
            initialization != 'random' or joint_supervision or history['loss_weight'] != 0):
        raise ValueError('Coarse-only memory requires random initialization and no residual objectives')
    expected_tokens = 0 if mode == 'none' else 2 if residual_enabled else 1
    if not model['freeze_vision'] or model['memory_tokens'] != expected_tokens:
        raise ValueError('Use frozen vision and memory_tokens=0 for none, 1 for coarse-only, or 2 for full')
    if model.get('memory_interface_version') != MEMORY_INTERFACE_VERSION:
        raise ValueError('Use memory_interface_version=separate_capped_v2; legacy interfaces are incompatible')
    for name in ('readout_rms_cap', 'memory_token_rms_cap'):
        value = model.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a positive finite number')
    if mode == 'none' and history['loss_weight'] != 0:
        raise ValueError('The no-memory action baseline requires history.loss_weight=0')
    if model.get('random_image_augmentation') or sft.get('random_augmentation'):
        raise ValueError('Frozen visual caches require deterministic image preprocessing')
    if not sft['full_prefix_bptt']:
        raise ValueError('History loss must reach the complete small-memory merge graph')
    if model['action_horizon'] != 20 or not 1 <= cfg['benchmark']['replan_steps'] <= 20:
        raise ValueError('Invalid prediction/execution horizons')
    for key in ('slots', 'coarse_dim', 'residual_dim', 'query_dim', 'action_dim', 'hidden_dim'):
        if not isinstance(memory[key], int) or memory[key] < 1:
            raise ValueError(f'Invalid memory dimension {key}')
    if memory['merge_rule'] not in {'ward', 'shortest', 'cosine'}:
        raise ValueError('Unknown adjacent merge rule')
    if memory['storage_dtype'] not in {'float32', 'bfloat16'}:
        raise ValueError('Unsupported persistent storage dtype')
    if memory.get('writer_version') != WRITER_VERSION:
        raise ValueError('Use writer_version=bounded_count_convex_v2; legacy writer semantics are incompatible')
    validate_writer_bound(memory.get('writer_bound'), memory['storage_dtype'])
    if history['image_grid'] < 2 or history['image_grid'] % 2:
        raise ValueError('Spatial teacher needs an even square image grid')
    if not history['camera_indices'] or len(set(history['camera_indices'])) != len(history['camera_indices']):
        raise ValueError('Select distinct camera slots')
    if history['max_queries'] < 1 or history['loss_weight'] < 0 or history['teacher_scale_floor'] <= 0:
        raise ValueError('Nonnegative history weight and positive sampling/normalization settings required')
    if history.get('query_sampling') != 'uniform_temporal_systematic':
        raise ValueError('History query_sampling must be uniform_temporal_systematic')
    if history.get('feature_kind') != 'fixed_spatial_projection':
        raise ValueError('Schema 4 requires direct fixed spatial features')
    if history.get('objective') not in {'residual', 'reconstruction'}:
        raise ValueError('Historical objective must be residual or reconstruction')
    if history.get('reader_mode', 'full') not in {'full', 'coarse', 'time'}:
        raise ValueError('Unknown historical reader control')
    if history.get('relative_energy_floor', 0) <= 0:
        raise ValueError('Historical relative energy threshold must be positive')
    if set(cfg['stages']) != {'memory', 'action'}:
        raise ValueError('Schema 4 has exactly memory and action training stages')
    if any(cfg['stages'][stage]['steps'] < 1 for stage in ('memory', 'action')):
        raise ValueError('Training budgets must be positive')
    memory_stage = cfg['stages']['memory']
    prefix_minimum = memory_stage.get('min_prefix_decisions')
    if prefix_minimum is not None and (type(prefix_minimum) is not int or prefix_minimum <= memory['slots']):
        raise ValueError('stages.memory.min_prefix_decisions must exceed memory.slots')
    required_step = memory_stage.get('action_checkpoint_step')
    if required_step is not None and (type(required_step) is not int or required_step != memory_stage['steps']):
        raise ValueError('stages.memory.action_checkpoint_step must match the full memory budget')
    minimum = cfg['stages']['memory'].get('min_validation_gain', .01)
    if (isinstance(minimum, bool) or not isinstance(minimum, (int, float))
            or not math.isfinite(minimum) or not 0 <= minimum < 1):
        raise ValueError('stages.memory.min_validation_gain must be finite in [0, 1)')
    if memory['coarse_dim'] % (4*len(history['camera_indices'])):
        raise ValueError('Teacher feature width must divide evenly over spatial regions')
    if sft['microbatch_size'] != 1 or sft['global_batch_size'] < 1:
        raise ValueError('Use one prefix per microbatch and positive global batch')
    if sft['parallelism'] not in {'fsdp_full_shard', 'fsdp_shard_grad_op'}:
        raise ValueError('Unknown FSDP strategy')
    if sft['gradient_sync'] not in {'every_microbatch', 'optimizer_step'}:
        raise ValueError('Unknown gradient synchronization policy')
    if sft.get('input_pipeline', 'cached_prefetch') not in {'cached_prefetch', 'synchronous'}:
        raise ValueError('Unknown history input pipeline')
    if sft.get('prefetch_depth', 1) < 1:
        raise ValueError('CPU prefetch depth must be positive')
    if sft.get('vision_batch_size', 1) != 1:
        raise ValueError('History SFT preserves single-frame frozen visual encoding')
    if sft['reference_world_size']*sft['gradient_accumulation_steps'] != sft['global_batch_size']:
        raise ValueError('Reference world size and accumulation must match the global batch')
    if any(sft[key] <= 0 for key in ('backbone_lr', 'memory_lr', 'interface_lr')):
        raise ValueError('Backbone, pretrained memory, and interface learning rates must be positive')
    if not sft['backbone_train_patterns']:
        raise ValueError('Explicit action/multimodal parameter patterns are required')
    validate_retention(cfg['training'])
    validate_action_validation(cfg)


def resolve_stage(cfg, stage, tasks=None):
    cfg = copy.deepcopy(cfg)
    if stage != 'action':
        raise ValueError('Action training stage must be action; feature/adapt/joint were removed')
    cfg['sft'].update(cfg['stages'][stage])
    cfg['sft']['stage'] = stage
    chosen = cfg['benchmark']['full_tasks'] if tasks is None else tasks
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(cfg['benchmark']['full_tasks']):
        raise ValueError('Invalid training tasks')
    cfg['training']['tasks'] = sorted(chosen)
    validate_config(cfg)
    return cfg


def memory_kwargs(cfg, prefix_dim=2048):
    import torch
    m, h = cfg['memory'], cfg['history']
    return dict(prefix_dim=prefix_dim, state_dim=8, coarse_dim=m['coarse_dim'],
                residual_dim=m['residual_dim'], query_dim=m['query_dim'], read_dim=m['action_dim'],
                hidden_dim=m['hidden_dim'], slots=m['slots'], image_grid=h['image_grid'],
                camera_indices=tuple(h['camera_indices']), storage_dtype=getattr(torch, m['storage_dtype']),
                merge_rule=m['merge_rule'], time_scale=m['time_scale'],
                writer_version=m['writer_version'], writer_bound=m['writer_bound'],
                teacher_seed=h['teacher_seed'], teacher_scale_floor=h['teacher_scale_floor'],
                objective=h['objective'], reader_mode=h.get('reader_mode', 'full'),
                relative_energy_floor=h['relative_energy_floor'],
                residual_enabled=m.get('residual_enabled', True))


def build_memory(cfg, prefix_dim=2048):
    from .core import HistoryMemory
    from .initialization import initialize_named_linears
    memory = HistoryMemory(**memory_kwargs(cfg, prefix_dim))
    initialize_named_linears(memory, cfg, 'memory',
        zero_linear_names=('codec.writer.ratio', 'codec.writer.mix', 'decoder.gate'))
    return memory


def architecture_signature(cfg):
    """Reject checkpoints with a different representation or supervision route."""
    signature = dict(schema_version=cfg['schema_version'], memory=copy.deepcopy(cfg['memory']),
                action_interface={key: copy.deepcopy(cfg['model'].get(key)) for key in
                                  ('memory_interface_version', 'readout_rms_cap', 'memory_token_rms_cap')},
                action_memory_mode=cfg['sft'].get('memory_mode', 'full'),
                history={k: copy.deepcopy(cfg['history'][k]) for k in
                         ('feature_kind', 'objective', 'reader_mode', 'image_grid',
                          'camera_indices', 'teacher_seed', 'teacher_scale_floor',
                          'relative_energy_floor')})
    if 'ablation' in cfg:
        signature['ablation_parameter_initialization'] = cfg['ablation']['parameter_initialization']
    return signature
