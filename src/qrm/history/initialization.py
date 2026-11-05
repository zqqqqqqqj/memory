"""Optional paired initialization for newly launched controlled ablations."""
import hashlib

import torch
from torch import nn


def initialize_named_linears(module, cfg, namespace, *, zero_linear_names=()):
    """Keep same-name/same-shape layers identical across residual widths.

    Each Linear gets its own CPU RNG stream. Teacher buffers and the official
    backbone are never reset. Legacy configurations retain their initialization.
    """
    scheme = cfg.get('ablation', {}).get('parameter_initialization')
    if scheme is None:
        return
    if scheme != 'named_linear_v1':
        raise ValueError('Unknown ablation parameter initialization')
    seed = torch.initial_seed()
    with torch.random.fork_rng(devices=[]):
        for name, layer in module.named_modules():
            if not isinstance(layer, nn.Linear):
                continue
            if layer.weight.device.type != 'cpu':
                raise ValueError('Ablation initialization must precede moving modules to CUDA')
            digest = hashlib.sha256(repr((seed, namespace, name)).encode()).digest()
            torch.random.default_generator.manual_seed(int.from_bytes(digest[:8], 'little') % (2**63-1))
            layer.reset_parameters()
        # Preserve the writer's count gates and the decoder's neutral query gate.
        for name in zero_linear_names:
            layer = module.get_submodule(name)
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
