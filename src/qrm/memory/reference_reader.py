"""Freeze the official action backbone before selecting trainable layers."""

def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module
