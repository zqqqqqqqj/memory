"""A fixed, spatially resolved visual teacher; fitting statistics is offline only.

The projection is a seeded orthogonal baseline, not an event recognizer. Frozen
backbone features are pooled into four regions per real camera before projection.
Missing camera slots never contribute. No future, action, or simulator state is
accepted by this interface.
"""
import math
from contextlib import contextmanager
import threading

import torch
from torch import nn


# The CUDA matmul precision setting is process-wide. Serialize our temporary
# overrides so overlapping fixed-projection calls restore the original setting.
_PROJECTION_PRECISION_LOCK = threading.RLock()


@contextmanager
def _projection_precision(device_type):
    """CPU-cache and CUDA-online projection both use ordinary FP32 matmul.

    OpenPI globally selects ``high``, which permits TF32 even when autocast is
    disabled. This local override affects only the fixed projection, restores
    the exact previous setting (including ``medium``), and does not change the
    backbone's BF16 policy. Cross-device FP32 reduction order may still differ;
    this is a precision contract, not a promise of bitwise CPU/CUDA equality.
    """
    if device_type != "cuda":
        yield
        return
    with _PROJECTION_PRECISION_LOCK:
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            yield
        finally:
            torch.set_float32_matmul_precision(previous)


def spatial_regions(visual, mask, *, image_grid=16, camera_indices=(0, 1)):
    """Return [batch, cameras*4, channels] and a region-validity mask.

    OpenPI image tokens are camera-major, then row-major in each square grid.
    A padded third camera is allowed, but the selected two camera indices are
    explicit; silently pooling all tokens would discard their spatial identity.
    """
    if image_grid < 2 or image_grid % 2:
        raise ValueError("image_grid must be an even integer >= 2")
    if not camera_indices or len(set(camera_indices)) != len(camera_indices):
        raise ValueError("camera_indices must be nonempty and unique")
    if visual.ndim != 3 or mask.shape != visual.shape[:2]:
        raise ValueError("Expected visual [B,N,D] and mask [B,N]")
    per_camera = image_grid * image_grid
    if visual.shape[1] % per_camera:
        raise ValueError("Visual tokens must contain complete camera grids")
    cameras = visual.shape[1] // per_camera
    if min(camera_indices) < 0 or max(camera_indices) >= cameras:
        raise ValueError("Selected camera index is absent from visual tokens")
    with torch.autocast(device_type=visual.device.type, enabled=False):
        x = visual.float().reshape(len(visual), cameras, image_grid, image_grid, -1)
        valid = mask.bool().reshape(len(visual), cameras, image_grid, image_grid)
        half = image_grid // 2
        result, validity = [], []
        for camera in camera_indices:
            for row in (0, half):
                for col in (0, half):
                    values = x[:, camera, row:row+half, col:col+half]
                    weights = valid[:, camera, row:row+half, col:col+half]
                    count = weights.sum((1, 2))
                    # where, rather than multiplication, also masks padded NaNs.
                    values = torch.where(weights[..., None], values, 0.)
                    result.append(values.sum((1, 2)) / count.clamp_min(1)[:, None])
                    validity.append(count > 0)
        return torch.stack(result, 1), torch.stack(validity, 1)


class FixedSpatialTeacher(nn.Module):
    def __init__(self, prefix_dim=2048, image_grid=16, camera_indices=(0, 1),
                 region_dim=32, seed=0, scale_floor=.01):
        super().__init__()
        if not 0 < region_dim <= prefix_dim:
            raise ValueError("region_dim must be in [1, prefix_dim]")
        if not math.isfinite(scale_floor) or scale_floor <= 0:
            raise ValueError("scale_floor must be finite and positive")
        self.prefix_dim, self.image_grid = prefix_dim, image_grid
        self.camera_indices = tuple(camera_indices)
        self.region_dim = region_dim
        self.output_dim = len(self.camera_indices) * 4 * region_dim
        self.scale_floor, self.seed = float(scale_floor), int(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        projection, _ = torch.linalg.qr(torch.randn(prefix_dim, region_dim, generator=generator))
        self.register_buffer("projection", projection)
        self.register_buffer("mean", torch.zeros(self.output_dim))
        self.register_buffer("scale", torch.ones(self.output_dim))
        self.register_buffer("raw_std", torch.zeros(self.output_dim))
        self.register_buffer("fitted", torch.tensor(False))
        self.register_buffer("fit_count", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("fit_valid_count", torch.zeros(self.output_dim, dtype=torch.int64))

    @torch.no_grad()
    def _project(self, visual, mask):
        if visual.shape[-1] != self.prefix_dim:
            raise ValueError("Teacher visual feature width differs from prefix_dim")
        regions, valid = spatial_regions(visual, mask, image_grid=self.image_grid,
                                         camera_indices=self.camera_indices)
        with torch.autocast(device_type=visual.device.type, enabled=False), _projection_precision(visual.device.type):
            projected = regions @ self.projection.float()
        if not torch.isfinite(projected).all():
            raise ValueError("Nonfinite valid visual features")
        return projected.flatten(1), valid[..., None].expand(-1, -1, self.region_dim).flatten(1)

    @torch.no_grad()
    def raw(self, visual, mask):
        return self._project(visual, mask)[0]

    @torch.no_grad()
    def fit_normalization(self, raw_targets, valid=None):
        """Fit once on explicitly selected TRAIN frames, then persist in checkpoint.

        The caller owns train/validation split enforcement and records provenance.
        Refitting a loaded teacher is rejected, preventing moving targets.
        """
        if self.fitted.item():
            raise RuntimeError("Teacher normalization is already fitted; create a new teacher to refit")
        if not torch.is_tensor(raw_targets):
            raw_targets = torch.cat(list(raw_targets), 0)
        if raw_targets.ndim != 2 or raw_targets.shape[1] != self.output_dim or len(raw_targets) < 2:
            raise ValueError("Need at least two finite training targets [N, output_dim]")
        if valid is None:
            valid = torch.ones_like(raw_targets, dtype=torch.bool)
        if valid.shape == (len(raw_targets), len(self.camera_indices)*4):
            valid = valid[..., None].expand(-1, -1, self.region_dim).flatten(1)
        if valid.shape != raw_targets.shape or valid.dtype != torch.bool or valid.device != raw_targets.device:
            raise ValueError("Expected matching boolean normalization validity mask")
        if not torch.isfinite(raw_targets[valid]).all():
            raise ValueError("Need finite valid training targets")
        count = valid.sum(0)
        if ((count > 0) & (count < 2)).any():
            raise ValueError("Every observed target dimension requires at least two training frames")
        values = torch.where(valid, raw_targets.detach().double(), 0.)
        mean = values.sum(0) / count.clamp_min(1)
        variance = torch.where(valid, (values-mean).square(), 0.).sum(0) / count.clamp_min(1)
        std = variance.sqrt()
        if std.max() <= 1e-8:
            raise ValueError("Teacher targets have no variation across training frames")
        self.mean.copy_(mean.to(self.mean))
        self.raw_std.copy_(std.to(self.raw_std))
        self.scale.copy_(std.clamp_min(self.scale_floor).to(self.scale))
        self.fit_count.fill_(len(values))
        self.fit_valid_count.copy_(count.to(self.fit_valid_count))
        self.fitted.fill_(True)

    @torch.no_grad()
    def normalize_raw(self, raw, valid):
        """Normalize cached FP32 projections with the identical online contract.

        ``valid`` may be per dimension or per spatial region. The validity mask
        is required so a missing camera cannot acquire a normalization offset.
        """
        if not self.fitted.item():
            raise RuntimeError("Fit teacher normalization on training frames before use")
        if raw.ndim != 2 or raw.shape[1] != self.output_dim:
            raise ValueError("Expected raw fixed projections [N, output_dim]")
        if valid.shape == (len(raw), len(self.camera_indices)*4):
            valid = valid[..., None].expand(-1, -1, self.region_dim).flatten(1)
        if valid.shape != raw.shape or valid.dtype != torch.bool or valid.device != raw.device:
            raise ValueError("Expected matching boolean projection validity mask")
        if raw.device != self.mean.device or not torch.isfinite(raw[valid]).all():
            raise ValueError("Raw projections must be finite on valid dimensions and share teacher device")
        with torch.autocast(device_type=raw.device.type, enabled=False):
            normalized = (raw.float() - self.mean.float()) / self.scale.float()
            return torch.where(valid, normalized, 0.).detach()

    @torch.no_grad()
    def forward(self, visual, mask):
        if not self.fitted.item():
            raise RuntimeError("Fit teacher normalization on training frames before use")
        return self.normalize_raw(*self._project(visual, mask))
