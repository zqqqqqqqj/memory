"""Bounded schema-4 deployment using the same fixed features as training."""
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from qrm.evaluation.manifest import sha256
from qrm.joint.model import frozen_visual
from qrm.memory.stream_codec import MemoryState
from qrm.normalization import checkpoint_norm_stats
from qrm.policies.pi05_bridge import Pi05Bridge
from .joint import HistoryJointModel, configure_backbone, current_prefix_only


def _validated_action_norm_stats(checkpoint):
    """A complete action EMA must bind its actual 8D/7D demonstration normalization."""
    path, digest = checkpoint.get('norm_stats'), checkpoint.get('norm_stats_sha256')
    if (not isinstance(path, str) or not path or not isinstance(digest, str) or len(digest) != 64
            or any(char not in '0123456789abcdef' for char in digest.lower())):
        raise ValueError('Complete action EMA requires norm_stats and norm_stats_sha256 metadata')
    verified = checkpoint_norm_stats(checkpoint)
    try:
        document = json.loads(Path(verified).read_text())['norm_stats']
        arrays = {}
        for name, width in (('state', 8), ('actions', 7)):
            arrays[name] = {}
            for key in ('mean', 'std', 'q01', 'q99'):
                values = document[name][key]
                if (not isinstance(values, list) or len(values) != width
                        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values)):
                    raise ValueError(f'Normalization {name}/{key} must contain {width} numeric values')
                tensor = torch.tensor(values, dtype=torch.float32)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f'Normalization {name}/{key} must be finite in FP32')
                arrays[name][key] = tensor
            if (arrays[name]['std'] < 0).any() or (arrays[name]['q99'] < arrays[name]['q01']).any():
                raise ValueError(f'Normalization {name} has invalid scale or quantile order')
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError('Complete action EMA has malformed 8D state / 7D action normalization') from error
    return verified


def _action_autocast(device):
    """Match the action training forward while retaining FP32 master/EMA weights."""
    device_type = torch.device(device).type
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16,
                          enabled=device_type == 'cuda')


def _prepare_ema_load(joint, model_state, cfg):
    """Restore selected-layer dtype BEFORE copying complete checkpoint weights.

    OpenPI constructs selected transformer layers in BF16. Loading their FP32
    EMA first would silently round small learned updates. Reuse the exact
    training selection, then finish deployment in frozen evaluation mode.
    """
    configure_backbone(joint.backbone, cfg['sft']['backbone_train_patterns'])
    for name, parameter in joint.backbone.named_parameters():
        if parameter.requires_grad:
            value = model_state.get('backbone.'+name)
            if value is None or value.dtype != torch.float32:
                raise ValueError(f'Complete action EMA requires FP32 trained parameter: {name}')
    disable_checkpointing = getattr(joint.backbone, 'gradient_checkpointing_disable', None)
    if disable_checkpointing is not None:
        disable_checkpointing()


class HistoryPolicy:
    def __init__(self, joint, sampling_seed=11):
        self.joint, self.bridge = joint, joint.bridge
        self.sampling_seed = sampling_seed
        self.state, self.active, self.metadata = MemoryState(), False, None

    def reset(self, metadata):
        self.state = MemoryState()
        self.metadata = dict(metadata)
        torch.manual_seed((self.sampling_seed+int(metadata.get('seed', 0))*1000003) % (2**63-1))
        self.active = True

    @torch.no_grad()
    def act(self, raw, env_step):
        if not self.active:
            raise RuntimeError('reset required before policy query')
        if set(raw) != {'observation/image', 'observation/wrist_image', 'observation/state', 'prompt'}:
            raise ValueError('Policy observation schema must be strictly whitelisted')
        observation, _ = self.bridge.transform(raw)
        prep = self.joint.backbone._preprocess_observation(observation, train=False)
        visual = frozen_visual(self.bridge, prep)
        language = self.joint.language(prep)
        prefix = self.joint.prefix(visual, language, prep[3])
        prepared = SimpleNamespace(observation=observation, preprocessing=prep, prefix=prefix)
        if self.joint.memory_mode == 'none':
            with _action_autocast(self.bridge.device):
                return self.bridge.infer(prepared, current_prefix_only, None)
        proprio = torch.as_tensor(raw['observation/state'], dtype=torch.float32, device=self.bridge.device)[None]
        memory = self.joint.memory
        feature = memory.visual(*visual)
        self.state = memory.codec.append(self.state, feature[0], env_step).detach()
        query = memory.current_query(feature, language, prep[3], proprio)[0]
        coarse, residual = memory.read(self.state, query)
        with _action_autocast(self.bridge.device):
            return self.bridge.infer(prepared, self.joint.receiver, (coarse[None], residual[None]))

    def finish(self, result):
        self.active = False

    def request(self, payload):
        if payload['op'] == 'reset':
            self.reset(payload['metadata'])
            return {'ok': True}
        if payload['op'] == 'act':
            return {'actions': self.act(payload['observation'], int(payload['env_step']))}
        if payload['op'] == 'finish':
            self.finish(payload['result'])
            return {'ok': True}
        raise ValueError(f"Unknown operation {payload['op']}")


def load_history_policy(checkpoint, *, device='cuda:0', payload=None):
    ckpt = payload if payload is not None else torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if (ckpt.get('schema_version') != 4 or ckpt.get('stage') != 'history_sft'
            or ckpt.get('history_stage') != 'action' or ckpt.get('weights') != 'complete_ema'):
        raise ValueError('Schema-4 deployment requires complete action-stage history SFT EMA weights')
    cfg = ckpt['config']
    from .config import validate_config
    validate_config(cfg)
    norm_stats = _validated_action_norm_stats(ckpt)
    base = Path(ckpt['backbone'])
    if sha256(base/'model.safetensors') != ckpt['backbone_sha256']:
        raise ValueError('Base checkpoint changed')
    torch.manual_seed(ckpt['seed'])
    bridge = Pi05Bridge(base, device=device, norm_stats=norm_stats,
                       action_horizon=cfg['model']['action_horizon'])
    joint = HistoryJointModel(bridge, cfg).to(device)
    _prepare_ema_load(joint, ckpt['model'], cfg)
    joint.load_state_dict(ckpt['model'], strict=True)
    joint.requires_grad_(False)
    joint.eval()
    if joint.memory_mode == 'full' and not joint.memory.teacher.fitted.item():
        raise ValueError('Checkpoint has no fitted frozen historical teacher')
    return HistoryPolicy(joint, ckpt['seed']), cfg, ckpt
