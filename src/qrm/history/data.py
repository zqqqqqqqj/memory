"""Episode-isolated, complete causal prefixes of frozen visual observations.

Visual and compact indices contain paths and provenance, never policy or
recurrent states. Old visual caches remain read-only after verification.
"""
from __future__ import annotations

import hashlib
import io
import json
import copy
from pathlib import Path
import random
from typing import Any

import torch


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def cache_key(episode_id: str, time: int) -> str:
    return hashlib.sha256(repr((episode_id, int(time))).encode()).hexdigest() + '.pt'


def _physical_identity(record: dict) -> tuple:
    source = record.get('source_record', record)
    if source.get('format') == 'lerobot_v3' or 'source_episode_index' in source:
        return ('lerobot_v3', source.get('root', ''), source.get('data_path', ''),
                source.get('dataset_from_index'), source.get('source_episode_index'))
    return ('episode_file', source.get('path', record['episode_id']))


def validate_index(document: dict, *, check_files: bool = True) -> None:
    if document.get('schema') != 1 or document.get('kind') != 'history_visual_index':
        raise ValueError('Expected history_visual_index schema 1; run scripts/prepare_history_cache.py')
    stride = document.get('stride', 0)
    if not isinstance(stride, int) or stride < 1:
        raise ValueError('History index stride must be a positive integer')
    ids, identities, physical = set(), set(), set()
    files_seen = set()
    splits = set()
    if not document.get('episodes'):
        raise ValueError('History index has no episodes')
    for record in document['episodes']:
        split = record['split']
        if split not in ('train', 'validation'):
            raise ValueError('History training index must exclude test episodes')
        splits.add(split)
        identity = (record['task_id'], record['seed'])
        physical_id = _physical_identity(record)
        if record['episode_id'] in ids or identity in identities or physical_id in physical:
            raise ValueError('Duplicate episode identity / physical source or split leakage')
        ids.add(record['episode_id']); identities.add(identity); physical.add(physical_id)
        times = record['times']
        if times != list(range(0, record['length'], stride)):
            raise ValueError('Index must contain the complete ordered decision grid, without truncation')
        if len(record['cache_files']) != len(times):
            raise ValueError('Cache files and times differ in length')
        for time, name in zip(times, record['cache_files'], strict=True):
            path = Path(name)
            if not path.is_absolute():
                raise ValueError('History cache paths must be absolute')
            if path.name != cache_key(record['episode_id'], time):
                raise ValueError('Cache file does not match episode/time identity')
            if str(path) in files_seen:
                raise ValueError('A frozen frame cannot be shared across episode identities')
            files_seen.add(str(path))
            if check_files and not path.is_file():
                raise FileNotFoundError(f'Missing frozen frame {path}. Re-run scripts/prepare_history_cache.py '
                                        'with the same --out and --encode-missing; partial prefixes are forbidden.')
    if splits != {'train', 'validation'}:
        raise ValueError('Both train and held-out validation episodes are required')


def read_index(path: str | Path, *, check_files: bool = True) -> dict:
    document = json.loads(Path(path).read_text())
    validate_index(document, check_files=check_files)
    source = document.get('source_manifest')
    if source:
        if file_sha256(source['path']) != source['sha256']:
            raise ValueError('Source episode manifest changed after frozen-cache export')
        original = json.loads(Path(source['path']).read_text())
        original_records = {r['episode_id']: r for r in original['episodes']}
        for record in document['episodes']:
            row = original_records.get(record['episode_id'])
            if row is None or any(record[k] != row[k] for k in ('split', 'task_id', 'seed', 'length')):
                raise ValueError('History episode identity/split disagrees with the source manifest')
            # Match the immutable episode boundaries/video pointers as well as its label.
            exported_source = record.get('source_record', {})
            if any(exported_source.get(k) != v for k, v in row.items() if k != 'path'):
                raise ValueError('History source record changed from the original demonstration')
    return document


def load_visual(path: str | Path, device='cpu', *, expected_sha256=None) -> tuple[torch.Tensor, torch.Tensor]:
    source = path
    if expected_sha256 is not None:
        # Verify exactly the bytes subsequently deserialized, with one disk read.
        contents = Path(path).read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected_sha256:
            raise ValueError(f'Current visual checksum differs from fixed feature source: {path}')
        source = io.BytesIO(contents)
    values = torch.load(source, map_location='cpu', weights_only=True)
    if not isinstance(values, (tuple, list)) or len(values) != 2:
        raise ValueError(f'Invalid frozen visual cache: {path}')
    visual, mask = values
    if (visual.ndim != 3 or visual.shape[0] != 1 or mask.shape != visual.shape[:2]
            or not visual.is_floating_point() or mask.dtype != torch.bool
            or not torch.isfinite(visual).all()):
        raise ValueError(f'Invalid frozen visual/mask shapes or values: {path}')
    return visual.to(device), mask.to(device)


class HistoryDataset:
    """Keep index metadata in RAM; load only the selected frame/prefix from disk."""
    def __init__(self, index: str | Path, split: str, *, check_files=True):
        if split not in ('train', 'validation'):
            raise ValueError('Only train/validation splits are allowed')
        self.index_path = Path(index).resolve()
        self.document = read_index(index, check_files=check_files)
        self.records = [r for r in self.document['episodes'] if r['split'] == split]
        self.split = split
        self.index_sha256 = file_sha256(index)
        self.by_task: dict[int, list[dict]] = {}
        for record in self.records:
            self.by_task.setdefault(record['task_id'], []).append(record)

    def sample_record(self, rng: random.Random) -> dict:
        task = rng.choice(sorted(self.by_task))
        return rng.choice(self.by_task[task])

    def sample_prefix(self, rng: random.Random, *, min_decisions=2) -> tuple[dict, int]:
        eligible = {task: [r for r in records if len(r['times']) >= min_decisions]
                    for task, records in self.by_task.items()}
        eligible = {task: records for task, records in eligible.items() if records}
        if not eligible:
            raise ValueError(f'No {self.split} episode has at least {min_decisions} decisions')
        task = rng.choice(sorted(eligible))
        record = rng.choice(eligible[task])
        end = rng.randrange(min_decisions - 1, len(record['times']))
        return record, end

    @staticmethod
    def prefix(record: dict, end_index: int, device='cpu') -> tuple[list, list[int]]:
        if not 0 <= end_index < len(record['times']):
            raise ValueError('Prefix endpoint outside the episode')
        return ([load_visual(p, device) for p in record['cache_files'][:end_index + 1]],
                record['times'][:end_index + 1])


class IndexedVisualCache:
    """Read-only adapter usable by joint SFT, without mutating reused caches."""
    def __init__(self, index: str | Path):
        self.document = read_index(index)
        self.entries = {(r['episode_id'], t): p for r in self.document['episodes']
                        for t, p in zip(r['times'], r['cache_files'], strict=True)}

    def lookup(self, identity: tuple):
        try:
            path = self.entries[identity]
        except KeyError as exc:
            raise KeyError(f'Unexported episode/time {identity}; regenerate the complete history index') from exc
        return load_visual(path)

    def get(self, identity, compute, device):
        # Deliberately never invoke compute: training cannot mutate dataset provenance.
        return tuple(value.to(device) for value in self.lookup(identity))


def load_teacher_initialization(index, teacher, *, document=None):
    """Load immutable normalization; this artifact is not a trained checkpoint."""
    from .compact_cache import load_teacher_initialization as load
    return load(index, teacher, document=document)


def load_prefix(record, end_index, teacher, device='cpu'):
    """Read an already indexed compact episode; Dataset adds reusable validation."""
    if not 0 <= end_index < len(record['times']):
        raise ValueError('Prefix endpoint outside the episode')
    if file_sha256(record['feature_file']) != record['feature_sha256']:
        raise ValueError('Compact feature checksum mismatch')
    payload = torch.load(record['feature_file'], map_location='cpu', weights_only=True)
    metadata, raw, valid = payload['metadata'], payload['raw'], payload['valid']
    if (any(metadata.get(k) != record[k] for k in ('episode_id', 'split', 'task_id', 'seed', 'times', 'length'))
            or raw.dtype != torch.float32 or raw.shape != (len(record['times']), teacher.output_dim)
            or valid.dtype != torch.bool or valid.shape != raw.shape or not torch.isfinite(raw).all()):
        raise ValueError('Compact episode identity, shape or dtype mismatch')
    teacher_device = teacher.projection.device
    return teacher.normalize_raw(raw[:end_index+1].to(teacher_device),
                                 valid[:end_index+1].to(teacher_device)).to(device)


class FeatureDataset(HistoryDataset):
    """Episode-isolated normalized fixed features with one bounded CPU cache."""
    def __init__(self, index, split, *, check_files=True, document=None):
        from .compact_cache import EpisodeFeatureReader, read_feature_index
        if split not in ('train', 'validation'):
            raise ValueError('Only train/validation splits are allowed')
        self.index_path = Path(index).resolve()
        self.document = read_feature_index(index, check_files=check_files, document=document)
        self.index_sha256 = self.document.index_sha256
        self.records = [r for r in self.document['episodes'] if r['split'] == split]
        self.split, self.by_task = split, {}
        for record in self.records:
            self.by_task.setdefault(record['task_id'], []).append(record)
        self.reader = EpisodeFeatureReader(self.document)

    def load_prefix(self, record, end_index, teacher, device='cpu'):
        if record['split'] != self.split or record not in self.records:
            raise ValueError('Compact episode is outside this dataset split')
        return self.reader.prefix(record, end_index, teacher, device)

    def load_frame(self, record, frame_index, teacher, device='cpu'):
        return self.load_prefix(record, frame_index, teacher, device)[frame_index]


class IndexedFeatureCache:
    """CPU fixed histories plus original current-frame tokens for joint SFT.

    The private teacher copy keeps prefetch threads off CUDA and prevents input
    preparation from mutating a loaded model's projection or normalization.
    """
    def __init__(self, index, teacher, *, document=None, check_files=True):
        from .compact_cache import EpisodeFeatureReader, read_feature_index
        self.document = read_feature_index(index, check_files=check_files, document=document)
        self.teacher = copy.deepcopy(teacher).cpu().eval().requires_grad_(False)
        original = {k: v.detach().cpu().clone() for k, v in self.teacher.state_dict().items()}
        fitted = bool(self.teacher.fitted.item())
        self.teacher_fit = load_teacher_initialization(index, self.teacher, document=self.document)
        if fitted and any(not torch.equal(original[k], value.cpu())
                          for k, value in self.teacher.state_dict().items()):
            raise ValueError('Loaded model teacher statistics differ from compact cache initialization')
        self.reader = EpisodeFeatureReader(self.document)
        self.records = {r['episode_id']: r for r in self.document['episodes']}
        self.entries = {(r['episode_id'], t): (r, i) for r in self.document['episodes']
                        for i, t in enumerate(r['times'])}

    def _entry(self, identity):
        try:
            return self.entries[identity]
        except KeyError as exc:
            raise KeyError(f'Unexported episode/time {identity}; regenerate the complete fixed feature index') from exc

    def lookup(self, identity):
        record, position = self._entry(identity)
        return self.reader.prefix(record, position, self.teacher)[position]

    def prefix(self, episode_id, end_time):
        record, position = self._entry((episode_id, end_time))
        return self.reader.prefix(record, position, self.teacher)

    def visual_path(self, identity):
        record, position = self._entry(identity)
        return record['cache_files'][position]

    def lookup_visual(self, identity):
        record, position = self._entry(identity)
        return load_visual(record['cache_files'][position],
                           expected_sha256=self.reader.visual_digest(record, position))

    def get(self, identity, compute, device):
        # Existing action-current-frame bridge; historical prefixes use prefix().
        return tuple(value.to(device) for value in self.lookup_visual(identity))
