"""Shared persistent C/Z state and serialization."""
from dataclasses import dataclass, replace
import torch

@dataclass(frozen=True)
class Block:
    c: torch.Tensor
    z: torch.Tensor
    n: int
    start: int
    end: int


@dataclass(frozen=True)
class MemoryState:
    blocks: tuple[Block, ...] = ()

    def detach(self):
        return MemoryState(tuple(replace(b, c=b.c.detach(), z=b.z.detach()) for b in self.blocks))

    def packed(self, slots: int, dc: int, dz: int, *, device="cpu", dtype=None):
        """Actual fixed-size deployable tensors, including explicit metadata."""
        dtype = dtype or (self.blocks[0].c.dtype if self.blocks else torch.bfloat16)
        c = torch.zeros(slots, dc, dtype=dtype, device=device)
        z = torch.zeros(slots, dz, dtype=dtype, device=device)
        meta = torch.zeros(slots, 3, dtype=torch.int64, device=device)
        valid = torch.zeros(slots, dtype=torch.bool, device=device)
        if len(self.blocks) > slots:
            raise ValueError("State exceeds capacity")
        for i, b in enumerate(self.blocks):
            c[i], z[i] = b.c.detach(), b.z.detach()
            meta[i] = torch.tensor([b.n, b.start, b.end], device=device)
            valid[i] = True
        return {"coarse": c, "residual": z, "metadata": meta, "valid": valid}


def state_nbytes(packed: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in packed.values())


def unpack_state(packed) -> MemoryState:
    blocks = []
    for i in range(len(packed["valid"])):
        if packed["valid"][i]:
            n, start, end = packed["metadata"][i].tolist()
            blocks.append(Block(packed["coarse"][i].clone(), packed["residual"][i].clone(), n, start, end))
    return MemoryState(tuple(blocks))
