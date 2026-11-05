"""Single-node torchrun support without changing the global optimizer batch."""
from contextlib import contextmanager
from datetime import timedelta
import hashlib
import os
import torch
import torch.distributed as dist


class DistributedContext:
    def __init__(self, device="cuda:0", *, initialize=True):
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(device)
        self.control = None
        if self.world_size > 1:
            if int(os.environ.get("LOCAL_WORLD_SIZE", str(self.world_size))) != self.world_size:
                raise ValueError("Only single-node shared-filesystem training is supported")
            if self.device.type == "cuda":
                self.device = torch.device("cuda", self.local_rank)
                torch.cuda.set_device(self.device)
            if initialize:
                timeout = timedelta(hours=12)
                dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo", timeout=timeout)
                self.control = dist.new_group(backend="gloo", timeout=timeout) if self.device.type == "cuda" else dist.group.WORLD

    @property
    def primary(self):
        return self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier(group=self.control)

    def broadcast(self, value):
        if self.world_size == 1:
            return value
        values = [value if self.primary else None]
        dist.broadcast_object_list(values, src=0, group=self.control)
        return values[0]

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        values = [None] * self.world_size if self.primary else None
        dist.gather_object(value, values, dst=0, group=self.control)
        return values

    @torch.no_grad()
    def sync_modules(self, *modules):
        if self.world_size > 1:
            for module in modules:
                for value in module.state_dict().values():
                    dist.broadcast(value, src=0)

    @torch.no_grad()
    def sum_gradients(self, parameters):
        """SUM disjoint loss terms; never divide the global gradient by GPU count."""
        if self.world_size == 1:
            return
        parameters = list(parameters)
        if any(p.dtype != torch.float32 or p.device != self.device for p in parameters):
            raise ValueError("Memory-module gradient reduction expects FP32 parameters on the local device")
        present = torch.tensor([p.grad is not None for p in parameters], dtype=torch.int32, device=self.device)
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        offset = 0
        for index, p in enumerate(parameters):
            size = p.numel()
            p.grad = flat[offset:offset+size].view_as(p).clone() if present[index].item() else None
            offset += size

    def sum_number(self, value):
        if self.world_size == 1:
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    def close(self):
        if self.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


@contextmanager
def loss_rng(seed, *identity, device):
    """Assign flow noise by data identity, independent of rank and GPU count."""
    digest = hashlib.sha256(repr((seed, identity)).encode()).digest()
    value = int.from_bytes(digest[:8], "little") % (2**63-1)
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(value)
        if devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(value)
        yield
