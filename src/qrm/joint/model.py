"""Shared frozen-vision/cache helpers used by schema-4 ablations."""
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class PrefixSample:
    visuals: list
    times: list
    observation: object
    preprocessing: tuple
    proprio: torch.Tensor
    actions: torch.Tensor
    identity: tuple


@torch.no_grad()
def frozen_visual(bridge, prep):
    embeddings, masks = [], []
    for image, mask in zip(prep[0], prep[1], strict=True):
        with torch.autocast(device_type=bridge.device.type, dtype=torch.bfloat16,
                            enabled=bridge.device.type == 'cuda'):
            visual = bridge.model.paligemma_with_expert.embed_image(image)
        embeddings.append(visual.detach())
        masks.append(mask[:, None].expand(*visual.shape[:2]))
    return torch.cat(embeddings, 1), torch.cat(masks, 1)


class VisualCache:
    """CPU/disk cache contains only f_i and validity, never P_x/Q/C/Z/LLM outputs."""
    def __init__(self, root, version):
        self.root = Path(root)/version
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, identity, compute, device):
        import hashlib, os
        key = hashlib.sha256(repr(identity).encode()).hexdigest()
        file = self.root/(key+'.pt')
        if file.exists():
            return tuple(t.to(device) for t in torch.load(file, map_location='cpu', weights_only=True))
        values = compute()
        # Atomic independent writes are safe when ranks sample the same frame.
        temp = file.with_suffix(f'.{os.getpid()}.tmp')
        torch.save(tuple(v.detach().cpu() for v in values), temp)
        temp.replace(file)
        return tuple(v.to(device) for v in values)
