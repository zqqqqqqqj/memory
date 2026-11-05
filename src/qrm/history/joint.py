"""Schema-4 fixed-feature memory and the actual OpenPI action loss share one graph."""
import math
import re
from pathlib import Path

import torch
from torch import nn

from qrm.memory.stream_codec import MemoryState
from qrm.policies.memory_receiver import injected_prefix
from .config import architecture_signature, build_memory
from .initialization import initialize_named_linears


def current_prefix_only(prefix, pad_mask, attention_mask, memory):
    """Keep the actual visual/language prefix unchanged for independent no-memory training."""
    if memory is not None:
        raise ValueError('The no-memory baseline must not receive a memory readout')
    return prefix, pad_mask, attention_mask


def cap_rms(values, maximum):
    """Apply an upper RMS bound per vector; never enlarge small or zero signals."""
    values = values.float()
    # Scale before the norm so finite, large readouts do not overflow squaring.
    magnitude = values.abs().amax(dim=-1, keepdim=True).clamp_min(1.)
    rms = torch.linalg.vector_norm(values/magnitude, dim=-1, keepdim=True)
    rms = (rms/math.sqrt(values.shape[-1]))*magnitude
    return values*(maximum/rms.clamp_min(maximum))


class SeparateMemoryReceiver(nn.Module):
    """Two named readouts, two independent lifts, no coarse+residual addition."""
    version = 'separate_capped_v2'

    def __init__(self, read_dim, prefix_dim, hidden=128, *, readout_rms_cap=4., memory_token_rms_cap=1.,
                 residual_enabled=True):
        super().__init__()
        for name, value in (('readout_rms_cap', readout_rms_cap), ('memory_token_rms_cap', memory_token_rms_cap)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive')
        self.readout_rms_cap = float(readout_rms_cap)
        self.memory_token_rms_cap = float(memory_token_rms_cap)
        self.residual_enabled = residual_enabled
        self.coarse_lift = nn.Sequential(nn.Linear(read_dim, hidden), nn.GELU(), nn.Linear(hidden, prefix_dim))
        self.residual_lift = nn.Sequential(nn.Linear(read_dim, hidden, bias=False), nn.GELU(),
                                           nn.Linear(hidden, prefix_dim, bias=False))
        # Retained state keys for explicit backbone-only/architecture compatibility.
        # V2 does not divide by calibration estimates or amplify near-zero signals.
        self.register_buffer('coarse_scale', torch.ones(()))
        self.register_buffer('residual_scale', torch.ones(()))

    def forward(self, prefix, pad_mask, attention_mask, readouts):
        coarse, residual = readouts
        coarse = self.coarse_lift(cap_rms(coarse, self.readout_rms_cap))
        # Leave room for the final BF16/FP32 rounding so the materialized token
        # retains the configured bound, including a learned coarse-lift bias.
        bound = self.memory_token_rms_cap/(1.+torch.finfo(prefix.dtype).eps)
        values = [cap_rms(coarse, bound)]
        if self.residual_enabled:
            residual = self.residual_lift(cap_rms(residual, self.readout_rms_cap))
            values.append(cap_rms(residual, bound))
        tokens = torch.stack(values, dim=1).to(prefix.dtype)
        valid = torch.ones((len(prefix), len(values)), dtype=pad_mask.dtype, device=prefix.device)
        causal = torch.zeros((len(prefix), len(values)), dtype=attention_mask.dtype, device=prefix.device)
        return torch.cat((prefix, tokens), 1), torch.cat((pad_mask, valid), 1), torch.cat((attention_mask, causal), 1)


def configure_backbone(backbone, patterns):
    """Freeze vision/projector/head and opt in to explicit backbone parameter names."""
    compiled = [re.compile(pattern) for pattern in patterns]
    counts = [0]*len(compiled)
    backbone.requires_grad_(False)
    for name, param in backbone.named_parameters():
        if any(part in {'vision_tower', 'multi_modal_projector', 'lm_head'} for part in name.split('.')):
            continue
        selected = [i for i, pattern in enumerate(compiled) if pattern.search(name)]
        if selected:
            param.requires_grad_(True)
            param.data = param.data.float()
            for i in selected:
                counts[i] += param.numel()
    if not any(counts):
        raise ValueError('No backbone parameters matched configured train patterns')
    if any(count == 0 for count in counts):
        missing = [pattern for pattern, count in zip(patterns, counts, strict=True) if count == 0]
        raise ValueError(f'Backbone train patterns matched no parameters: {missing}')
    if hasattr(backbone, 'gradient_checkpointing_enable'):
        backbone.gradient_checkpointing_enable()
    backbone.train()
    for module in backbone.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.
    vision = getattr(getattr(getattr(backbone, 'paligemma_with_expert', None), 'paligemma', None), 'vision_tower', None)
    if vision is not None:
        vision.eval()
    return dict(zip(patterns, counts, strict=True))


class HistoryJointModel(nn.Module):
    def __init__(self, bridge, cfg):
        super().__init__()
        self.bridge, self.cfg = bridge, cfg
        self.backbone = bridge.model
        self.memory_mode = cfg['sft'].get('memory_mode', 'full')
        if self.memory_mode not in {'none', 'full'}:
            raise ValueError('memory_mode must be none or full')
        self.memory = build_memory(cfg, bridge.prefix_dim)
        self.receiver = SeparateMemoryReceiver(cfg['memory']['action_dim'], bridge.prefix_dim,
            readout_rms_cap=cfg['model'].get('readout_rms_cap', 4.),
            memory_token_rms_cap=cfg['model'].get('memory_token_rms_cap', 1.),
            residual_enabled=self.memory.residual_enabled)
        initialize_named_linears(self.receiver, cfg, 'receiver')
        if self.memory_mode == 'none':
            self.memory.requires_grad_(False)
            self.receiver.requires_grad_(False)

    def language(self, prep):
        tokens = self.backbone.paligemma_with_expert.embed_language_tokens(prep[2])
        return tokens*math.sqrt(tokens.shape[-1])

    def prefix(self, visual, language, lang_mask):
        emb, mask = (v.to(language.device) for v in visual)
        prefix = torch.cat((emb.to(language.dtype), language), 1)
        padding = torch.cat((mask, lang_mask), 1)
        return prefix, padding, torch.zeros_like(padding)

    def replay(self, visuals, times, device):
        if not visuals or len(visuals) != len(times):
            raise ValueError('A nonempty complete causal prefix is required')
        state, targets = MemoryState(), []
        for visual, time in zip(visuals, times, strict=True):
            values = tuple(v.to(device) for v in visual)
            x = self.memory.visual(*values)
            state = self.memory.codec.append(state, x[0], time)
            with torch.no_grad():
                targets.append(self.memory.teacher(*values)[0])
        return state, x, torch.stack(targets)

    def replay_features(self, features, times, device):
        """Replay all fixed descriptors; only the small merge graph is differentiable."""
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape != (len(times), self.memory.coarse_dim) or not len(times)
                or features.dtype != torch.float32 or not torch.isfinite(features).all()):
            raise ValueError('Expected a complete finite FP32 fixed-feature prefix [T,D]')
        targets = features.detach().to(device)
        state = MemoryState()
        for x, timestamp in zip(targets, times, strict=True):
            state = self.memory.codec.append(state, x, timestamp)
        return state, targets[-1:], targets

    def forward(self, sample):
        language = self.language(sample.preprocessing)
        if self.memory_mode == 'none':
            current_visual = sample.current_visual if hasattr(sample, 'current_visual') else sample.visuals[-1]
            prefix = self.prefix(current_visual, language, sample.preprocessing[3])
            with injected_prefix(self.backbone, prefix, sample.preprocessing, current_prefix_only, None):
                with torch.autocast(device_type=sample.actions.device.type, dtype=torch.bfloat16,
                                    enabled=sample.actions.device.type == 'cuda'):
                    per_element = self.backbone(sample.observation, sample.actions).float()
            loss = per_element.mean()
            zero = loss.detach().new_zeros(())
            return loss, dict(prefix_decisions=1, merged=0, action_loss=loss.detach(),
                action_loss_7d=per_element[..., :7].mean().detach(),
                action_padding_loss=per_element[..., 7:].mean().detach() if per_element.shape[-1] > 7 else zero,
                history_loss=zero, history_history_queries=0, coarse_rms=zero, residual_rms=zero)
        if hasattr(sample, 'features'):
            prefix = self.prefix(sample.current_visual, language, sample.preprocessing[3])
            state, x, targets = self.replay_features(sample.features, sample.times, sample.proprio.device)
        else:
            # Read-only reference path for equivalence checks; production loads
            # compact historical features and only the current full visual token.
            prefix = self.prefix(sample.visuals[-1], language, sample.preprocessing[3])
            state, x, targets = self.replay(sample.visuals, sample.times, sample.proprio.device)
        query = self.memory.current_query(x, language, sample.preprocessing[3], sample.proprio)[0]
        coarse, residual = self.memory.read(state, query)
        with injected_prefix(self.backbone, prefix, sample.preprocessing, self.receiver,
                             (coarse[None], residual[None])):
            with torch.autocast(device_type=sample.actions.device.type, dtype=torch.bfloat16,
                                enabled=sample.actions.device.type == 'cuda'):
                per_element = self.backbone(sample.observation, sample.actions).float()
                action_loss = per_element.mean()
                actual_control_loss = per_element[..., :7].mean()
                padding_loss = (per_element[..., 7:].mean() if per_element.shape[-1] > 7
                                else per_element.new_zeros(()))
        if self.cfg['history'].get('joint_supervision', True):
            history_loss, metrics = self.memory.history_loss(
                state, sample.times, targets, max_queries=self.cfg['history']['max_queries'], tensor_metrics=True)
        else:
            history_loss = action_loss.new_zeros(())
            metrics = dict(history_queries=0)
        loss = action_loss+self.cfg['history']['loss_weight']*history_loss
        info = dict(prefix_decisions=len(sample.times), merged=int(any(b.n > 1 for b in state.blocks)),
                    action_loss=action_loss.detach(), action_loss_7d=actual_control_loss.detach(),
                    action_padding_loss=padding_loss.detach(),
                    history_loss=history_loss.detach(), coarse_rms=coarse.detach().square().mean().sqrt(),
                    residual_rms=residual.detach().square().mean().sqrt())
        info.update({f'history_{k}': v.detach() if isinstance(v, torch.Tensor) else v
                     for k, v in metrics.items() if isinstance(v, (int, float, torch.Tensor))})
        return loss, info

    def small_parameters(self):
        yield from self.memory.parameters()
        yield from self.receiver.parameters()

    def configure_stage(self, stage):
        if stage != 'action':
            raise ValueError('Schema-4 action training has one action stage; adapt/joint were removed')
        if self.memory_mode == 'none':
            self.memory.requires_grad_(False)
            self.receiver.requires_grad_(False)
            return
        if self.memory.reader_mode != 'full':
            raise ValueError('Action training requires the full memory; coarse/time are independent pretraining controls')
        self.memory.requires_grad_(True)
        self.memory.teacher.requires_grad_(False)
        for name in ('coarse_control', 'time_control', 'control_heads'):
            module = getattr(self.memory, name, None)
            if module is not None:
                module.requires_grad_(False)
        unused_head = ('reconstruction_head' if self.memory.objective == 'residual' else 'history_head')
        getattr(self.memory, unused_head).requires_grad_(False)
        self.receiver.requires_grad_(True)
        if not self.cfg['history'].get('joint_supervision', True):
            for name in ('time_query', 'history_head', 'reconstruction_head'):
                getattr(self.memory, name).requires_grad_(False)
        if not self.memory.residual_enabled:
            self.memory.codec.writer.requires_grad_(False)
            for name in ('residual_skip', 'z_in', 'gate', 'z_out'):
                getattr(self.memory.decoder, name).requires_grad_(False)
            self.receiver.residual_lift.requires_grad_(False)


def optimizer_groups(model, cfg):
    """Stable parameter groups, created once before the first action update."""
    original = getattr(model, 'module', model)
    backbone = {id(p) for p in original.backbone.parameters()}
    pretrained = ({id(p) for module in original.memory.history_train_modules('full')
                   for p in module.parameters()}
                  if original.memory_mode == 'full' and original.memory.residual_enabled else set())
    rates = dict(backbone=cfg['sft']['backbone_lr'], memory=cfg['sft']['memory_lr'],
                 interface=cfg['sft']['interface_lr'])
    groups = {name: [] for name in rates}
    for parameter in model.parameters():
        if parameter.requires_grad:
            name = 'backbone' if id(parameter) in backbone else 'memory' if id(parameter) in pretrained else 'interface'
            groups[name].append(parameter)
    return [dict(params=params, lr=rates[name], peak_lr=rates[name], name=name)
            for name, params in groups.items() if params]


def load_memory_initialization(joint, checkpoint):
    if joint.memory_mode == 'none':
        raise ValueError('The independent no-memory baseline must not load a memory checkpoint')
    if joint.cfg['sft'].get('memory_initialization', 'pretrained') == 'random':
        raise ValueError('Random memory initialization must not load a memory checkpoint')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('schema_version') != 4 or payload.get('stage') != 'memory':
        raise ValueError('Action training requires a schema-4 stage=memory pretraining checkpoint')
    if architecture_signature(payload['config']) != architecture_signature(joint.cfg):
        raise ValueError('Memory pretraining architecture differs from joint configuration')
    joint.memory.load_state_dict(payload['memory'], strict=True)
    if not joint.memory.teacher.fitted.item():
        raise ValueError('Memory pretraining teacher has not been fitted')
    if payload.get('reader_mode', 'full') != 'full':
        raise ValueError('Independent coarse/time controls cannot initialize the full action model')
    return payload


def load_initial_checkpoint(joint, checkpoint, *, norm_stats_sha256, stage, backbone_sha256=None,
                            memory_checkpoint_sha256=None, history_index_sha256=None, backbone_only=False):
    """New-stage weight transfer; deliberately never transfers optimizer/RNG."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if payload.get('weights') != 'complete_ema' or payload.get('norm_stats_sha256') != norm_stats_sha256:
        raise ValueError('Initialization needs complete EMA and matching normalization')
    if backbone_sha256 is not None and payload.get('backbone_sha256') != backbone_sha256:
        raise ValueError('Initialization backbone differs from cached frozen visual encoder')
    legacy = payload.get('schema_version') == 2 and payload.get('stage') == 'joint_sft'
    action = (payload.get('schema_version') == 4 and payload.get('stage') == 'history_sft'
              and payload.get('history_stage') == 'action')
    if stage != 'action' or not (legacy or action):
        raise ValueError('Action initialization requires schema-4 action EMA or explicit legacy backbone-only weights')
    if backbone_only:
        state = {name.removeprefix('backbone.'): value for name, value in payload['model'].items()
                 if name.startswith('backbone.')}
        joint.backbone.load_state_dict(state, strict=True)
        return 'legacy_backbone_only' if legacy else 'action_backbone_only'
    if legacy:
        raise ValueError('Legacy initialization requires explicit backbone_only=True')
    if architecture_signature(payload['config']) != architecture_signature(joint.cfg):
        raise ValueError('Action checkpoint memory architecture changed')
    for name, expected in (('memory_checkpoint_sha256', memory_checkpoint_sha256),
                           ('history_index_sha256', history_index_sha256)):
        if expected is not None and payload.get(name) != expected:
            raise ValueError(f'Action checkpoint was trained from a different {name}')
    joint.load_state_dict(payload['model'], strict=True)
    return 'action_complete_model'
