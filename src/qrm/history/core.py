"""Shared deterministic write/read path for historical supervision and actions.

Only merge selection is detached. Every selected continuous merge retains its
autograd graph through BF16 storage casts. Persistent state is still C/Z plus
metadata; training may hold a longer graph, deployment never does.
"""
import math

import torch
from torch import nn

from qrm.memory.stream_codec import Block, MemoryState
from .config import WRITER_VERSION, validate_writer_bound
from .teacher import FixedSpatialTeacher


class CurrentQuery(nn.Module):
    """Only the action query is learned; visual writes use the fixed teacher."""
    def __init__(self, prefix_dim=2048, state_dim=8, coarse_dim=256, query_dim=256,
                 hidden_dim=256):
        super().__init__()
        self.language_project = nn.Linear(prefix_dim, query_dim)
        self.query = nn.Sequential(nn.Linear(coarse_dim + query_dim + state_dim, hidden_dim),
                                   nn.GELU(), nn.Linear(hidden_dim, query_dim))

    def current_query(self, x, language, language_mask, proprio):
        with torch.autocast(device_type=x.device.type, enabled=False):
            weights = language_mask.float()
            pooled = (language.float() * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)
            return self.query(torch.cat((x.float(), self.language_project(pooled), proprio.float()), -1))


class ResidualWriter(nn.Module):
    """Count-adaptive convex mixture of carry and a bounded ordered candidate.

    Starting with singleton z=0, every coordinate stays within [-B, B]: both
    the child carry and tanh candidate are in that interval and sigmoid supplies
    convex weights. The candidate gate is scaled by twice the smaller child's
    observation fraction. Equal-sized merges retain the original gate; adding
    one frame to n stored frames starts with carry coefficient (n/(n+1))**2,
    rather than repeatedly halving the old state. This avoids exponential loss
    along the initial carry path, but neither a state bound nor a nonzero
    gradient guarantees useful memory. Both child graphs remain differentiable.
    """
    def __init__(self, dc=256, dz=32, hidden=256, bound=1.0):
        super().__init__()
        self.bound = validate_writer_bound(bound)
        self.version = WRITER_VERSION
        self.contrast = nn.Linear(dc, dz, bias=False)
        self.value = nn.Linear(dc + 2*dz, hidden, bias=False)
        self.ratio = nn.Linear(2, hidden)
        self.out = nn.Linear(hidden, dz, bias=False)
        self.mix = nn.Linear(2, dz)
        nn.init.zeros_(self.ratio.weight)
        nn.init.zeros_(self.ratio.bias)
        nn.init.zeros_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)

    def forward(self, left, right):
        with torch.autocast(device_type=left.c.device.type, enabled=False):
            contrast = left.c.float() - right.c.float()
            rho = contrast.new_tensor([left.n, right.n]) / (left.n + right.n)
            carry = rho[0]*left.z.float() + rho[1]*right.z.float()
            value = torch.cat((contrast, left.z.float(), right.z.float()))
            hidden = torch.nn.functional.gelu(self.value(value)) * (1 + self.ratio(rho).tanh())
            candidate = self.bound * (self.contrast(contrast) + self.out(hidden)).tanh()
            mix = (2 * rho.min()) * self.mix(rho).sigmoid()
            return (1 - mix)*carry + mix*candidate


class HistoryCodec(nn.Module):
    def __init__(self, slots=8, coarse_dim=256, residual_dim=32, hidden_dim=256,
                 storage_dtype=torch.bfloat16, merge_rule="ward",
                 writer_version=WRITER_VERSION, writer_bound=1.0, residual_enabled=True):
        super().__init__()
        if slots < 1 or merge_rule not in {"ward", "shortest", "cosine"}:
            raise ValueError("Invalid memory slots or merge rule")
        if storage_dtype not in {torch.float32, torch.bfloat16}:
            raise ValueError("Supported storage dtypes are float32 and bfloat16")
        if writer_version != WRITER_VERSION:
            raise ValueError(f'Only writer_version={WRITER_VERSION} is supported; legacy writers require a separate project')
        self.writer_version = writer_version
        self.writer_bound = validate_writer_bound(writer_bound, str(storage_dtype).removeprefix('torch.'))
        self.slots, self.dc, self.dz = slots, coarse_dim, residual_dim
        self.storage_dtype, self.merge_rule = storage_dtype, merge_rule
        self.residual_enabled = residual_enabled
        self.writer = ResidualWriter(coarse_dim, residual_dim, hidden_dim, bound=self.writer_bound)

    @torch.no_grad()
    def _merge_index(self, blocks):
        pairs = list(zip(blocks[:-1], blocks[1:], strict=True))
        counts = [left.n + right.n for left, right in pairs]
        if self.merge_rule == "shortest":
            scores = counts
        elif self.merge_rule == "ward":
            scores = torch.stack([(left.c.float() - right.c.float()).square().sum()
                                  * (left.n*right.n/(left.n+right.n)) for left, right in pairs]).tolist()
        else:
            scores = torch.stack([-torch.nn.functional.cosine_similarity(left.c.float(), right.c.float(), dim=0)
                                  for left, right in pairs]).tolist()
        return min(range(len(pairs)), key=lambda i: (scores[i], counts[i], i))

    def append(self, state, x, time):
        if x.shape != (self.dc,) or not torch.isfinite(x).all():
            raise ValueError("Expected a finite one-dimensional visual summary")
        if int(time) != time or time < 0 or (state.blocks and time <= state.blocks[-1].end):
            raise ValueError("Observation times must be nonnegative and strictly increasing")
        if len(state.blocks) > self.slots:
            raise ValueError("Input memory exceeds capacity")
        with torch.autocast(device_type=x.device.type, enabled=False):
            blocks = list(state.blocks) + [Block(x.to(self.storage_dtype),
                           x.new_zeros(self.dz if self.residual_enabled else 0, dtype=self.storage_dtype),
                           1, int(time), int(time))]
            if len(blocks) > self.slots:
                index = self._merge_index(blocks)
                left, right = blocks[index:index+2]
                n = left.n + right.n
                coarse = (left.c.float()*(left.n/n) + right.c.float()*(right.n/n)).to(self.storage_dtype)
                residual = (self.writer(left, right).to(self.storage_dtype) if self.residual_enabled
                            else x.new_empty(0, dtype=self.storage_dtype))
                blocks[index:index+2] = [Block(coarse, residual, n, left.start, right.end)]
            return MemoryState(tuple(blocks))

    def replay(self, features, times):
        state = MemoryState()
        for feature, time in zip(features, times, strict=True):
            state = self.append(state, feature, time)
            yield state


class SharedDecoder(nn.Module):
    """Global action attention and local history queries share residual values."""
    def __init__(self, dc=256, dz=32, dq=256, read_dim=128, hidden=256, time_scale=2500.,
                 residual_enabled=True):
        super().__init__()
        if not math.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("time_scale must be positive")
        self.dc, self.dz, self.dq, self.read_dim = dc, dz, dq, read_dim
        self.time_scale = time_scale
        self.residual_enabled = residual_enabled
        self.attention = nn.Sequential(nn.Linear(dq+dc+3, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.coarse_value = nn.Sequential(nn.Linear(dc+3, hidden), nn.GELU(), nn.Linear(hidden, read_dim))
        self.residual_skip = nn.Linear(dz, read_dim, bias=False)
        self.z_in = nn.Linear(dz, hidden, bias=False)
        self.gate = nn.Linear(dq+3, hidden)
        self.z_out = nn.Linear(hidden, read_dim, bias=False)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def residual_value(self, residual, query, tau):
        """Shared by local history queries and global action reads; zero z -> zero.

        Coarse content and attention never enter this value function. A local
        history query selects its stored block directly, whereas an action query
        learns attention across all blocks.
        """
        with torch.autocast(device_type=residual.device.type, enabled=False):
            context = torch.cat((query.float(), tau.float()), -1)
            hidden = self.z_in(residual.float()) * (1 + self.gate(context).tanh())
            return self.residual_skip(residual.float()) + self.z_out(torch.nn.functional.gelu(hidden))

    def block_times(self, state, *, device):
        now = state.blocks[-1].end
        return torch.tensor([[(now-block.start)/self.time_scale,
                              (now-block.end)/self.time_scale, math.log1p(block.n)]
                             for block in state.blocks], dtype=torch.float32, device=device)

    def forward(self, state, query, mode="full"):
        if mode not in {"full", "coarse", "time"}:
            raise ValueError("mode must be full, coarse, or time")
        if query.shape != (self.dq,):
            raise ValueError("Expected a single query [query_dim]")
        if not state.blocks:
            zero = query.new_zeros(self.read_dim, dtype=torch.float32)
            return zero, zero.clone()
        with torch.autocast(device_type=query.device.type, enabled=False):
            coarse = torch.stack([block.c for block in state.blocks]).float()
            residual = torch.stack([block.z for block in state.blocks]).float()
            if mode != "full":
                residual = torch.zeros_like(residual)
            if mode == "time":
                coarse = torch.zeros_like(coarse)
            tau = self.block_times(state, device=query.device)
            context = torch.cat((query.float().expand(len(coarse), -1), coarse, tau), -1)
            attention = self.attention(context).squeeze(-1).softmax(0)
            coarse_values = self.coarse_value(torch.cat((coarse, tau), -1))
            z_values = (self.residual_value(residual, query.expand(len(coarse), -1), tau)
                        if self.residual_enabled else torch.zeros_like(coarse_values))
            return ((attention[:, None]*coarse_values).sum(0),
                    (attention[:, None]*z_values).sum(0))


class HistoryMemory(nn.Module):
    def __init__(self, prefix_dim=2048, state_dim=8, coarse_dim=256, residual_dim=32,
                 query_dim=256, read_dim=128, hidden_dim=256, slots=8, image_grid=16,
                 camera_indices=(0, 1), storage_dtype=torch.bfloat16, merge_rule="ward",
                 time_scale=2500., teacher_seed=0, teacher_scale_floor=.01,
                 objective="residual", reader_mode="full", relative_energy_floor=1e-12,
                 writer_version=WRITER_VERSION, writer_bound=1.0, residual_enabled=True):
        super().__init__()
        camera_indices = tuple(camera_indices)
        if not camera_indices or coarse_dim % (4*len(camera_indices)):
            raise ValueError("coarse_dim must be divisible by 4 * camera_count")
        self.coarse_dim, self.query_dim, self.read_dim = coarse_dim, query_dim, read_dim
        if objective not in {"residual", "reconstruction"}:
            raise ValueError("Historical objective must be residual or reconstruction")
        if reader_mode not in {"full", "coarse", "time"}:
            raise ValueError("Historical reader_mode must be full, coarse, or time")
        if not math.isfinite(relative_energy_floor) or relative_energy_floor <= 0:
            raise ValueError("relative_energy_floor must be finite and positive")
        self.objective, self.reader_mode = objective, reader_mode
        self.residual_enabled = residual_enabled
        self.relative_energy_floor = relative_energy_floor
        self.time_scale = time_scale
        self.teacher = FixedSpatialTeacher(prefix_dim, image_grid, camera_indices,
                              coarse_dim//(4*len(camera_indices)), teacher_seed, teacher_scale_floor)
        self.feature = CurrentQuery(prefix_dim, state_dim, coarse_dim, query_dim, hidden_dim)
        self.codec = HistoryCodec(slots, coarse_dim, residual_dim, hidden_dim, storage_dtype, merge_rule,
                                  writer_version, writer_bound, residual_enabled)
        self.decoder = SharedDecoder(coarse_dim, residual_dim, query_dim, read_dim, hidden_dim, time_scale,
                                     residual_enabled)
        self.time_query = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, query_dim))
        # Both layers are bias-free: a zero stored residual cannot generate a
        # time-dependent reconstruction by passing through learned head biases.
        self.history_head = nn.Sequential(nn.Linear(read_dim, hidden_dim, bias=False), nn.GELU(),
                                          nn.Linear(hidden_dim, coarse_dim, bias=False))
        self.reconstruction_head = nn.Sequential(nn.Linear(2*read_dim, hidden_dim), nn.GELU(),
                                                 nn.Linear(hidden_dim, coarse_dim))
        # Independently trainable controls, not zero-z variants of the full model.
        # Their temporal metadata includes vision-dependent merge boundaries.
        self.control_heads = nn.ModuleDict({
            "coarse": nn.Sequential(nn.Linear(coarse_dim+6, hidden_dim), nn.GELU(),
                                    nn.Linear(hidden_dim, coarse_dim)),
            "time": nn.Sequential(nn.Linear(6, hidden_dim), nn.GELU(),
                                  nn.Linear(hidden_dim, coarse_dim)),
        })

    def visual(self, visual, mask):
        return self.teacher(visual, mask)

    def history_train_modules(self, mode=None):
        if not self.residual_enabled:
            raise ValueError('Coarse-only memory has no historical reconstruction objective')
        mode = self.reader_mode if mode is None else mode
        if mode in {"coarse", "time"}:
            return (self.control_heads[mode],)
        if mode != "full":
            raise ValueError("mode must be full, coarse, or time")
        modules = (self.codec.writer, self.decoder.residual_skip, self.decoder.z_in,
                   self.decoder.gate, self.decoder.z_out, self.time_query)
        if self.objective == "reconstruction":
            return modules + (self.decoder.coarse_value, self.reconstruction_head)
        return modules + (self.history_head,)

    def current_query(self, x, language, language_mask, proprio):
        return self.feature.current_query(x, language, language_mask, proprio)

    def read(self, state, query, mode="full"):
        return self.decoder(state, query, mode)

    def rollout(self, visuals, times):
        state, x = MemoryState(), None
        for (visual, mask), time in zip(visuals, times, strict=True):
            x = self.visual(visual, mask)
            if len(x) != 1:
                raise ValueError("Rollout accepts one episode prefix at a time")
            state = self.codec.append(state, x[0], time)
        if x is None:
            raise ValueError("Cannot reconstruct an empty prefix")
        return state, x

    def historical_block(self, state, target_time):
        if not state.blocks:
            raise ValueError("Cannot query empty history")
        now = state.blocks[-1].end
        if int(target_time) != target_time or target_time < state.blocks[0].start or target_time >= now:
            raise ValueError("Historical query must precede current time and lie in stored history")
        for index, block in enumerate(state.blocks):
            if block.start <= target_time <= block.end:
                if block.n <= 1:
                    raise ValueError("Historical query must belong to a merged block")
                return index, block
        raise ValueError("Historical query falls outside an observed block")

    def reconstruct(self, state, target_time, mode=None, *, objective=None, zero_residual=False):
        """Predict a block-local delta (default), or the reconstruction control.

        The query contains a target's local position, age, and block span. It has
        no target image, current image, or learned coarse reconstruction bypass
        in residual mode. Training supplies only real observed target times.
        """
        mode = self.reader_mode if mode is None else mode
        objective = self.objective if objective is None else objective
        if mode not in {"full", "coarse", "time"}:
            raise ValueError("mode must be full, coarse, or time")
        if objective not in {"residual", "reconstruction"}:
            raise ValueError("objective must be residual or reconstruction")
        index, block = self.historical_block(state, target_time)
        with torch.autocast(device_type=block.c.device.type, enabled=False):
            span = block.end-block.start
            local_time = block.c.new_tensor([(target_time-block.start)/max(span, 1),
                              (state.blocks[-1].end-target_time)/self.time_scale,
                              span/self.time_scale], dtype=torch.float32)
            tau = self.decoder.block_times(state, device=block.c.device)[index]
            if mode != "full":
                values = (block.c.detach().float(), local_time, tau) if mode == "coarse" else (local_time, tau)
                return self.control_heads[mode](torch.cat(values, -1))
            query = self.time_query(local_time)
            z = torch.zeros_like(block.z) if zero_residual else block.z
            residual = self.decoder.residual_value(z, query, tau)
            if objective == "residual":
                return self.history_head(residual)
            coarse = self.decoder.coarse_value(torch.cat((block.c.float(), tau), -1))
            return self.reconstruction_head(torch.cat((coarse, residual), -1))

    def history_loss(self, state, times, targets, **kwargs):
        from .supervision import history_loss
        return history_loss(self, state, times, targets, **kwargs)
