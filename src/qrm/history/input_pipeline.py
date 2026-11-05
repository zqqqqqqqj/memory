"""Ordered CPU inputs: compact past features plus the current full visual frame.

No frozen or trainable CUDA module runs in the producer. Every sampled prefix
starts at episode time zero; discarding lookahead cannot skip a training sample.
"""
from contextlib import closing
from collections import OrderedDict
import copy
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import types

import numpy as np
import torch

from qrm.data import decisions
from qrm.lerobot_v3 import episode_numeric, numeric_file
from qrm.history.data import load_visual
from qrm.joint.input_pipeline import PrefixRequests
from qrm.prefetch import PrefetchedInputs


class _NumericRAMCache:
    """Retain whole parsed tables, so episode views share one backing allocation."""
    def __init__(self, max_bytes=512 * 1024**2):
        self.max_bytes = max_bytes
        self.cache = OrderedDict()
        self.used_bytes = 0

    def __call__(self, path):
        info = os.stat(path)
        key = (path, info.st_dev, info.st_ino, info.st_size,
               info.st_mtime_ns, info.st_ctime_ns)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        # Bypass only the old one-file LRU, retaining its unchanged parser.
        values = numeric_file.__wrapped__(path)
        size = sum(value.nbytes for value in values.values())
        for old in list(self.cache):
            if old[0] == path:
                self.used_bytes -= self.cache.pop(old)[1]
        if size <= self.max_bytes:
            while self.cache and self.used_bytes + size > self.max_bytes:
                _, (_, removed) = self.cache.popitem(last=False)
                self.used_bytes -= removed
            self.cache[key] = (values, size)
            self.used_bytes += size
        return values


class _BoundedFeatureReader:
    """Expand only this reader's existing validated raw-feature LRU by byte budget."""
    def __init__(self, reader, max_bytes=512 * 1024**2):
        # Do not change the supplied IndexedFeatureCache or its original reader.
        reader = copy.copy(reader)
        reader.cache = OrderedDict(reader.cache)
        reader.source_digests = dict(reader.source_digests)
        self.reader, self.max_bytes = reader, max_bytes
        self.original_raw = reader.raw
        self.sizes = {}
        self.used_bytes = 0
        # The unchanged raw implementation still performs SHA, metadata and tensor
        # validation. Its count limit is replaced by the byte limit below.
        reader.max_episodes = sys.maxsize
        for name in reader.cache:
            self._account(name)
        reader.raw = self.raw

    def _account(self, name):
        values = self.reader.cache[name]
        digests = self.reader.source_digests[name]
        size = (sum(value.numel() * value.element_size() for value in values)
                + sys.getsizeof(values) + sys.getsizeof(digests)
                + sum(sys.getsizeof(value) for value in digests)
                + sys.getsizeof(name) + 256)
        self.sizes[name] = size
        self.used_bytes += size

    def raw(self, record):
        values = self.original_raw(record)
        name = record['feature_file']
        if name not in self.sizes:
            self._account(name)
        # Keep the current entry until visual_digest has consumed its checksums.
        # A single oversized episode remains bounded by its own checked file.
        while len(self.reader.cache) > 1 and self.used_bytes > self.max_bytes:
            old, _ = self.reader.cache.popitem(last=False)
            self.reader.source_digests.pop(old)
            self.used_bytes -= self.sizes.pop(old)
        return values


class _LocalVisualCache:
    """Disposable, byte-verified, shared local copies of immutable visual files.

    A miss reads the original exactly once. Filling is best-effort and never
    waits for another process's writer lock. A conservative persistent byte
    reservation bounds concurrent writes, including leftovers after a crash.
    """
    def __init__(self, root=None, max_bytes=850 * 1024**3,
                 min_free_bytes=256 * 1024**3):
        configured = root if root is not None else os.environ.get(
            'QRM_READ_CACHE_ROOT', f'/tmp/qrm-read-cache-{os.getuid()}')
        self.root = Path(configured)
        self.directory = self.root / 'visual-v1'
        self.max_bytes, self.min_free_bytes = max_bytes, min_free_bytes
        self.enabled = True
        self.stats = dict(hits=0, misses=0, corrupt=0, filled=0, skipped=0)
        try:
            self._private_directory(self.root)
            self._private_directory(self.directory)
        except OSError:
            self.enabled = False

    @staticmethod
    def _private_directory(path):
        try:
            path.mkdir(mode=0o700)
        except FileExistsError:
            pass
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise PermissionError(f'Unsafe local read cache directory: {path}')

    @staticmethod
    def _read_private(path):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                raise PermissionError(f'Unsafe local read cache file: {path}')
            return stream.read()

    @staticmethod
    def _sync_directory(path):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _reserve(self, size):
        ledger = self.directory / 'quota.json'
        try:
            value = json.loads(self._read_private(ledger))
            if (not isinstance(value, dict) or value.get('schema') != 1
                    or type(value.get('reserved_bytes')) is not int
                    or value['reserved_bytes'] < 0):
                return False
        except FileNotFoundError:
            # Never infer that old unaccounted files or temporary files are free.
            if any(path.name != '.write.lock' for path in self.directory.iterdir()):
                return False
            value = dict(schema=1, reserved_bytes=0)
        except (ValueError, TypeError):
            return False
        if value['reserved_bytes'] + size > self.max_bytes:
            return False
        free = os.statvfs(self.directory)
        if free.f_bavail * free.f_frsize < self.min_free_bytes + size:
            return False
        value['reserved_bytes'] += size
        fd, temporary = tempfile.mkstemp(prefix='.quota.', dir=self.directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(json.dumps(value, sort_keys=True).encode())
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, ledger)
            self._sync_directory(self.directory)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        return True

    def _fill(self, path, contents, expected):
        lock_fd = None
        temporary = None
        try:
            lock_fd = os.open(self.directory / '.write.lock',
                              os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            info = os.fstat(lock_fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                return
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # A competing reader may have published after our miss.
            try:
                self._private_existing_directory(path.parent)
                if hashlib.sha256(self._read_private(path)).hexdigest() == expected:
                    return
            except OSError:
                pass
            if not self._reserve(len(contents)):
                return
            self._private_directory(path.parent)
            fd, temporary = tempfile.mkstemp(prefix=f'.{expected}.', dir=path.parent)
            with os.fdopen(fd, 'wb') as stream:
                stream.write(contents)
                stream.flush()
                # Only the disposable replica is synchronized; source is read-only.
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
            self._sync_directory(path.parent)
            self.stats['filled'] += 1
        except (OSError, ValueError, TypeError):
            self.stats['skipped'] += 1
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
            if lock_fd is not None:
                os.close(lock_fd)

    def read(self, source, expected_sha256):
        if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
                or any(c not in '0123456789abcdef' for c in expected_sha256)):
            raise ValueError('Invalid expected visual SHA256')
        path = self.directory / expected_sha256[:2] / (expected_sha256 + '.pt')
        if self.enabled:
            try:
                # Checking the shard also refuses symlinked/foreign directories.
                self._private_existing_directory(path.parent)
                contents = self._read_private(path)
                if hashlib.sha256(contents).hexdigest() == expected_sha256:
                    self.stats['hits'] += 1
                    return contents
                self.stats['corrupt'] += 1
            except OSError:
                pass
        self.stats['misses'] += 1
        contents = Path(source).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected_sha256:
            raise ValueError(f'Current visual checksum differs from fixed feature source: {source}')
        if self.enabled:
            self._fill(path, contents, expected_sha256)
        return contents

    @staticmethod
    def _private_existing_directory(path):
        info = path.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise PermissionError(f'Unsafe local read cache directory: {path}')


class _RuntimeInputs:
    def __init__(self, cache):
        self.numeric_tables = _NumericRAMCache()
        # Reuse the original adapter's exact bytecode, including all episode,
        # frame, task, shape and finite checks, without changing module globals.
        bindings = dict(episode_numeric.__globals__, numeric_file=self.numeric_tables)
        self.episode_numeric = types.FunctionType(
            episode_numeric.__code__, bindings, episode_numeric.__name__,
            episode_numeric.__defaults__, episode_numeric.__closure__)
        self.feature_reader = _BoundedFeatureReader(cache.reader)
        self.visual = _LocalVisualCache()

    def prefix(self, cache, identity):
        record, position = cache._entry(identity)
        return self.feature_reader.reader.prefix(record, position, cache.teacher)

    def lookup_visual(self, cache, identity):
        record, position = cache._entry(identity)
        expected = self.feature_reader.reader.visual_digest(record, position)
        contents = self.visual.read(record['cache_files'][position], expected)
        # The original loader performs unchanged shape/dtype/finiteness checks.
        return load_visual(io.BytesIO(contents))


def _runtime_inputs(cache):
    if os.environ.get('QRM_INPUT_CACHE_DISABLE') == '1':
        return None
    state = getattr(cache, '_qrm_runtime_inputs', None)
    if state is None:
        state = _RuntimeInputs(cache)
        cache._qrm_runtime_inputs = state
    return state


@dataclass
class FixedPrefixSample:
    features: torch.Tensor
    current_visual: tuple
    times: list
    observation: object
    preprocessing: tuple
    proprio: torch.Tensor
    actions: torch.Tensor
    identity: tuple


@dataclass
class CPUFixedHistory:
    identity: tuple
    times: list
    features: torch.Tensor
    current_visual: tuple
    inputs: dict
    proprio: np.ndarray
    metrics: dict


def prepare_cpu_history(bridge, record, selected_time, stride, cache):
    if cache is None:
        raise ValueError('Schema-4 training requires a completed fixed-feature cache')
    if stride < 1 or not 0 <= selected_time < record['length'] or selected_time % stride:
        raise ValueError('Selected query outside the complete decision grid')
    start = time.monotonic()
    identity = (record['episode_id'], selected_time)
    times = list(range(0, selected_time+1, stride))
    runtime = _runtime_inputs(cache)
    features = runtime.prefix(cache, identity) if runtime is not None else cache.prefix(*identity)
    if (features.device.type != 'cpu' or features.dtype != torch.float32
            or features.ndim != 2 or len(features) != len(times) or not torch.isfinite(features).all()):
        raise ValueError('Invalid CPU fixed-feature prefix')
    visual = runtime.lookup_visual(cache, identity) if runtime is not None else cache.lookup_visual(identity)
    if any(value.device.type != 'cpu' for value in visual):
        raise ValueError('The input producer must not access CUDA')
    # Fresh current state/prompt/action targets still use the audited OpenPI
    # transform. The complete past has already been encoded and verified.
    stream = (_cached_numeric_decision(record, bridge.horizon, selected_time, cache, visual)
              if record.get('format') == 'lerobot_v3' else decisions(record, bridge.horizon, stride))
    inputs = proprio = None
    with closing(stream):
        for timestamp, raw, actions in stream:
            if timestamp == selected_time:
                inputs = bridge.transform_cpu(raw, actions)
                proprio = np.array(raw['observation/state'], dtype=np.float32, copy=True)
                break
    if inputs is None or proprio is None or proprio.shape != (8,) or not np.isfinite(proprio).all():
        raise ValueError('Missing or invalid current observation at selected decision')
    return CPUFixedHistory(identity, times, features, visual, inputs, proprio,
                           dict(cpu_prepare_seconds=time.monotonic()-start,
                                cache_hits=1, cache_misses=0, prepared_image_decisions=1,
                                fixed_feature_frames=len(times),
                                fixed_feature_bytes=features.numel()*features.element_size()))


def materialize_history(bridge, history, cache=None):
    observation, normalized = bridge.transform_prepared(history.inputs)
    prep = bridge.model._preprocess_observation(observation, train=False)
    return FixedPrefixSample(history.features, history.current_visual, history.times,
                             observation, prep,
                             torch.as_tensor(history.proprio, device=bridge.device)[None],
                             normalized, history.identity)


def prefetch_histories(bridge, records, seed, first_index, stop_index, world, stride,
                       cache, capacity=1):
    requests = PrefixRequests(records, seed, first_index, stop_index, world, stride)
    return PrefetchedInputs(requests,
                            lambda record, timestamp: prepare_cpu_history(
                                bridge, record, timestamp, stride, cache), capacity=capacity)


def _cached_numeric_decision(record, horizon, selected_time, cache, visual):
    """Read numeric inputs against the verified, fixed visual cache.

    Numeric samples remain read and validated by the original adapter. RGB is
    present only for deterministic OpenPI transform structure; the verified
    cached visual is the sole image representation consumed by action forward.
    """
    bound = cache.records.get(record['episode_id'])
    if bound is None or bound['source_record'] != record:
        raise ValueError('Current numeric source differs from the verified visual cache source record')
    emb, mask = visual
    if emb.shape[:2] != (1, 768) or mask.shape != (1, 768):
        raise ValueError('Cached input path requires the pinned 3 camera x 16x16 token layout')
    expected = torch.cat((torch.ones(1, 512, dtype=torch.bool), torch.zeros(1, 256, dtype=torch.bool)), 1)
    if not torch.equal(mask.bool(), expected):
        raise ValueError('Cached image masks differ from the pinned front/wrist/padding contract')
    runtime = _runtime_inputs(cache)
    states, actions = (runtime.episode_numeric(record) if runtime is not None
                       else episode_numeric(record))
    target = actions[selected_time:min(len(actions), selected_time + horizon)]
    if len(target) < horizon:
        target = np.concatenate([target, np.repeat(target[-1:], horizon - len(target), axis=0)])
    raw = {'observation/image': np.zeros((256, 256, 3), dtype=np.uint8),
           'observation/wrist_image': np.zeros((256, 256, 3), dtype=np.uint8),
           'observation/state': states[selected_time], 'prompt': record['prompt']}
    yield selected_time, raw, target
