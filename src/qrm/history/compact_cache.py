"""Immutable, episode-contiguous fixed features and train-only normalization.

Preparation is a CPU data conversion, never an optimizer/training stage. Raw
projections are cached so normalization has one auditable train-only artifact.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tempfile

import torch

from .data import file_sha256, read_index, validate_index


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def tensor_sha256(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def teacher_spec(teacher):
    return dict(prefix_dim=teacher.prefix_dim, image_grid=teacher.image_grid,
                camera_indices=list(teacher.camera_indices), region_dim=teacher.region_dim,
                seed=teacher.seed, scale_floor=teacher.scale_floor, output_dim=teacher.output_dim)


def projection_source_signature():
    root = Path(__file__).parent
    return {name: file_sha256(root/name) for name in ('teacher.py', 'compact_cache.py', 'data.py')}


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path, value):
    _atomic(path, lambda stream: stream.write((json.dumps(value, indent=2)+'\n').encode()))


@contextmanager
def _preparation_lock(out):
    with (out/'prepare.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _episode_metadata(record, plan):
    return dict(schema=1, kind='history_raw_features', episode_id=record['episode_id'],
                split=record['split'], task_id=record['task_id'], seed=record['seed'],
                times=record['times'], length=record['length'], plan_sha256=_digest(plan),
                output_dim=plan['teacher']['output_dim'])


def _source_frames_digest(times, digests):
    content = hashlib.sha256()
    for time, digest in zip(times, digests, strict=True):
        content.update(json.dumps([time, digest], separators=(',', ':')).encode())
    return content.hexdigest()


def _check_episode(payload, record, plan):
    if not isinstance(payload, dict) or payload.get('metadata') != _episode_metadata(record, plan):
        raise ValueError('Compact episode identity/projection provenance mismatch')
    raw, valid = payload.get('raw'), payload.get('valid')
    shape = (len(record['times']), plan['teacher']['output_dim'])
    if (not torch.is_tensor(raw) or raw.dtype != torch.float32 or tuple(raw.shape) != shape
            or not torch.isfinite(raw).all() or not torch.is_tensor(valid)
            or valid.dtype != torch.bool or tuple(valid.shape) != shape):
        raise ValueError('Invalid compact feature dtype, shape, or values')
    digests = payload.get('source_frame_sha256')
    if (not isinstance(digests, list) or len(digests) != len(record['times'])
            or any(not isinstance(digest, str) or len(digest) != 64
                   or any(char not in '0123456789abcdef' for char in digest) for digest in digests)
            or payload.get('source_frames_sha256') != _source_frames_digest(record['times'], digests)):
        raise ValueError('Missing or inconsistent per-frame source visual content digests')
    if ('source_frames_sha256' in record
            and record['source_frames_sha256'] != payload['source_frames_sha256']):
        raise ValueError('Source visual content digests differ from compact index')
    return raw, valid


def _convert_episode(record, teacher, plan):
    raw, valid, digests = [], [], []
    for time, name in zip(record['times'], record['cache_files'], strict=True):
        # Hash and deserialize the SAME bytes: source verification does not add
        # another pass over hundreds of GB of full visual tokens.
        data = Path(name).read_bytes()
        values = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError(f'Invalid frozen visual cache: {name}')
        visual, mask = values
        if (visual.ndim != 3 or visual.shape[0] != 1 or mask.shape != visual.shape[:2]
                or not visual.is_floating_point() or mask.dtype != torch.bool
                or not torch.isfinite(visual).all()):
            raise ValueError(f'Invalid frozen visual/mask shapes or values: {name}')
        projected, validity = teacher._project(visual, mask)
        raw.append(projected.cpu().float()); valid.append(validity.cpu().bool())
        digests.append(hashlib.sha256(data).hexdigest())
    payload = dict(metadata=_episode_metadata(record, plan), raw=torch.cat(raw), valid=torch.cat(valid),
                   source_frame_sha256=digests,
                   source_frames_sha256=_source_frames_digest(record['times'], digests))
    _check_episode(payload, record, plan)
    return payload


def _selected_identities(records, *, samples, seed):
    if samples < 2:
        raise ValueError('Teacher fitting requires at least two requested train frames')
    candidates = [(r['episode_id'], index, time) for r in records if r['split'] == 'train'
                  for index, time in enumerate(r['times'])]
    if len(candidates) < 2:
        raise ValueError('Teacher fitting requires at least two available train frames')
    return random.Random(seed).sample(candidates, min(samples, len(candidates)))


def _selected_raw(records, selected, teacher, plan):
    positions = {}
    for destination, (episode, index, _) in enumerate(selected):
        positions.setdefault(episode, []).append((destination, index))
    raw = torch.empty(len(selected), teacher.output_dim, dtype=torch.float32)
    valid = torch.empty_like(raw, dtype=torch.bool)
    for record in records:
        locations = positions.get(record['episode_id'])
        if locations is None:
            continue
        payload = torch.load(record['feature_file'], map_location='cpu', weights_only=True)
        values, mask = _check_episode(payload, record, plan)
        destination, source = zip(*locations)
        raw[list(destination)] = values[list(source)]
        valid[list(destination)] = mask[list(source)]
    return raw, valid


def _fit_artifact(records, teacher, plan, out):
    selected = _selected_identities(records, samples=plan['normalization']['requested_samples'],
                                    seed=plan['normalization']['seed'])
    selected_ids = [(episode, time) for episode, _, time in selected]
    fit = dict(split='train', requested_samples=plan['normalization']['requested_samples'],
               samples=len(selected), seed=plan['normalization']['seed'],
               source_index_sha256=plan['source_index']['sha256'],
               selected_identities=selected_ids, selected_identity_sha256=_digest(selected_ids),
               episode_ids=sorted({episode for episode, _ in selected_ids}))
    path = out/'teacher_initialization.pt'
    receipt = path.with_suffix('.json')
    expected = dict(schema=1, kind='fixed_teacher_initialization', plan_sha256=_digest(plan),
                    teacher=plan['teacher'], projection_sha256=plan['projection_sha256'],
                    source_code=plan['source_code'])
    if path.exists():
        artifact = torch.load(path, map_location='cpu', weights_only=True)
        if any(artifact.get(k) != v for k, v in expected.items()):
            raise ValueError('Existing teacher initialization differs from preparation provenance')
        if any(artifact['fit'].get(k) != v for k, v in fit.items()):
            raise ValueError('Existing teacher initialization uses different train frame identities')
        expected_receipt = dict(sha256=file_sha256(path), plan_sha256=_digest(plan))
        if receipt.exists():
            if json.loads(receipt.read_text()) != expected_receipt:
                raise ValueError('Existing teacher initialization checksum changed')
        else:
            # Crash between atomic artifact and receipt publication: verify the
            # statistics again from small raw tensors, never original images.
            recovered = copy.deepcopy(teacher)
            recovered.fit_normalization(*_selected_raw(records, selected, recovered, plan))
            if any(not torch.equal(value, artifact['teacher_state'][key])
                   for key, value in recovered.state_dict().items()):
                raise ValueError('Unpublished teacher statistics differ from selected training frames')
            _atomic_json(receipt, expected_receipt)
        teacher.load_state_dict(artifact['teacher_state'], strict=True)
        _check_teacher(teacher, plan, len(selected))
        return path, artifact
    teacher.fit_normalization(*_selected_raw(records, selected, teacher, plan))
    fit.update(min_raw_std=float(teacher.raw_std.min()), median_raw_std=float(teacher.raw_std.median()),
               scale_floor=teacher.scale_floor,
               floor_dimensions=int((teacher.raw_std < teacher.scale_floor).sum()))
    artifact = dict(expected, fit=fit, teacher_state={k: v.cpu().clone() for k, v in teacher.state_dict().items()})
    _atomic(path, lambda stream: torch.save(artifact, stream))
    _atomic_json(receipt, dict(sha256=file_sha256(path), plan_sha256=_digest(plan)))
    return path, artifact


def _check_teacher(teacher, plan, count):
    if (teacher_spec(teacher) != plan['teacher'] or tensor_sha256(teacher.projection) != plan['projection_sha256']
            or not teacher.fitted.item() or teacher.fit_count.item() != count
            or not torch.isfinite(teacher.mean).all() or not torch.isfinite(teacher.scale).all()
            or (teacher.scale < teacher.scale_floor).any()):
        raise ValueError('Teacher normalization/projection state differs from immutable preparation')


def prepare_features(source_index, out, teacher, *, fit_frames=16384, seed=20261016):
    """Convert complete episodes, resume verified outputs, publish index LAST.

    This deliberately runs on CPU. Parallel training processes later read the
    same immutable files; preparation does not initialize CUDA, DDP or W&B.
    """
    if next(teacher.buffers()).device.type != 'cpu':
        raise ValueError('Fixed feature preparation is CPU-only')
    if teacher.fitted.item():
        raise ValueError('Preparation requires a fresh, unfitted fixed teacher')
    source_index, out = Path(source_index).resolve(), Path(out).resolve()
    if out == source_index.parent:
        raise ValueError('Use a separate output directory; original visual index must remain read-only')
    source = read_index(source_index, check_files=False)
    plan = dict(schema=1, kind='fixed_feature_preparation', source_index=dict(path=str(source_index),
                sha256=file_sha256(source_index)), teacher=teacher_spec(teacher),
                projection_sha256=tensor_sha256(teacher.projection), source_code=projection_source_signature(),
                normalization=dict(requested_samples=int(fit_frames), seed=int(seed), split='train'))
    _selected_identities(source['episodes'], samples=fit_frames, seed=seed)
    out.mkdir(parents=True, exist_ok=True)
    with _preparation_lock(out):
        plan_path = out/'preparation.json'
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise ValueError('Preparation provenance changed; use a new output directory')
        if not plan_path.exists():
            _atomic_json(plan_path, plan)
        if (out/'index.json').exists():
            completed = read_feature_index(out/'index.json')
            if completed['preparation'] != plan:
                raise ValueError('Published compact index differs from requested preparation')
            return completed
        records = []
        for number, original in enumerate(source['episodes'], 1):
            record = copy.deepcopy(original)
            path = out/'episodes'/(hashlib.sha256(record['episode_id'].encode()).hexdigest()+'.pt')
            receipt = path.with_suffix('.json')
            if path.exists():
                payload = torch.load(path, map_location='cpu', weights_only=True)
                _check_episode(payload, record, plan)
            else:
                payload = _convert_episode(record, teacher, plan)
                _atomic(path, lambda stream: torch.save(payload, stream))
            digest = file_sha256(path)
            expected_receipt = dict(sha256=digest, metadata=payload['metadata'])
            if receipt.exists() and json.loads(receipt.read_text()) != expected_receipt:
                raise ValueError('Previously prepared compact episode checksum changed')
            if not receipt.exists():
                _atomic_json(receipt, expected_receipt)
            record.update(feature_file=str(path), feature_sha256=digest,
                          source_frames_sha256=payload['source_frames_sha256'])
            records.append(record)
            _atomic_json(out/'preparation_status.json', dict(status='projecting', complete_episodes=number,
                         total_episodes=len(source['episodes']), source_index_sha256=plan['source_index']['sha256']))
            print(f'Fixed features: {number}/{len(source["episodes"])} complete episodes', flush=True)
        artifact_path, artifact = _fit_artifact(records, teacher, plan, out)
        document = dict(source, schema=1, kind='history_fixed_feature_index', episodes=records,
                        preparation=plan, teacher_initialization=dict(path=str(artifact_path),
                            sha256=file_sha256(artifact_path)),
                        feature_dtype='float32', validity_dtype='bool', feature_dim=teacher.output_dim)
        validate_feature_index(document, check_files=True)
        # A changed source cannot be published under a stale preparation hash.
        if file_sha256(source_index) != plan['source_index']['sha256']:
            raise ValueError('Source index changed during compact feature preparation')
        _atomic_json(out/'index.json', document)
        _atomic_json(out/'preparation_status.json', dict(status='complete', complete_episodes=len(records),
                     total_frames=sum(len(r['times']) for r in records),
                     normalization_samples=artifact['fit']['samples'], index_sha256=file_sha256(out/'index.json')))
        return document


class VerifiedFeatureDocument(dict):
    """One verified index shared by train/validation without repeating full scans."""
    def __init__(self, value, path, sha256, check_files):
        super().__init__(value)
        self.index_path, self.index_sha256, self.files_checked = Path(path).resolve(), sha256, check_files


def validate_feature_index(document, *, check_files=True):
    if document.get('schema') != 1 or document.get('kind') != 'history_fixed_feature_index':
        raise ValueError('Expected history_fixed_feature_index; run scripts/prepare_history_features.py')
    validate_index(dict(document, kind='history_visual_index'), check_files=False)
    plan = document['preparation']
    if (plan.get('kind') != 'fixed_feature_preparation' or plan.get('schema') != 1
            or plan['normalization']['split'] != 'train' or document.get('feature_dtype') != 'float32'
            or document.get('validity_dtype') != 'bool' or document['feature_dim'] != plan['teacher']['output_dim']):
        raise ValueError('Invalid fixed feature preparation metadata')
    if plan['source_code'] != projection_source_signature():
        raise ValueError('Fixed feature source code changed; regenerate cache in a new directory')
    source = plan['source_index']
    if file_sha256(source['path']) != source['sha256']:
        raise ValueError('Original visual index changed after compact export')
    original = read_index(source['path'], check_files=False)
    if any(document.get(k) != original.get(k) for k in ('stride', 'source_manifest', 'visual_provenance')):
        raise ValueError('Compact source preprocessing/manifest/stride metadata changed')
    originals = {record['episode_id']: record for record in original['episodes']}
    if len(originals) != len(document['episodes']):
        raise ValueError('Compact index must preserve all source train and validation episodes')
    paths = set()
    for record in document['episodes']:
        source_record = originals.get(record['episode_id'])
        if source_record is None or any(record.get(k) != v for k, v in source_record.items()):
            raise ValueError('Compact episode identity/split/grid differs from original visual index')
        path = Path(record['feature_file'])
        if not path.is_absolute() or str(path) in paths:
            raise ValueError('Compact feature files must be unique absolute paths per episode')
        paths.add(str(path))
        if check_files:
            if file_sha256(path) != record['feature_sha256']:
                raise ValueError('Compact feature checksum mismatch')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            _check_episode(payload, record, plan)
            if payload['source_frames_sha256'] != record['source_frames_sha256']:
                raise ValueError('Source content digest differs from compact episode')
    artifact = document['teacher_initialization']
    if not Path(artifact['path']).is_absolute() or file_sha256(artifact['path']) != artifact['sha256']:
        raise ValueError('Teacher initialization checksum mismatch')


def read_feature_index(path, *, check_files=True, document=None):
    path = Path(path).resolve()
    if document is not None:
        if (not isinstance(document, VerifiedFeatureDocument) or document.index_path != path
                or (check_files and not document.files_checked)):
            raise ValueError('Share only a previously verified compact index from the same path')
        return document
    value = json.loads(path.read_text())
    validate_feature_index(value, check_files=check_files)
    return VerifiedFeatureDocument(value, path, file_sha256(path), check_files)


def load_teacher_initialization(index, teacher, *, document=None):
    document = read_feature_index(index, check_files=False, document=document)
    plan = document['preparation']
    if teacher_spec(teacher) != plan['teacher'] or tensor_sha256(teacher.projection) != plan['projection_sha256']:
        raise ValueError('Configured fixed teacher differs from compact cache projection')
    artifact_info = document['teacher_initialization']
    if file_sha256(artifact_info['path']) != artifact_info['sha256']:
        raise ValueError('Teacher initialization checksum mismatch')
    artifact = torch.load(artifact_info['path'], map_location='cpu', weights_only=True)
    expected = dict(schema=1, kind='fixed_teacher_initialization', plan_sha256=_digest(plan),
                    teacher=plan['teacher'], projection_sha256=plan['projection_sha256'], source_code=plan['source_code'])
    if any(artifact.get(k) != v for k, v in expected.items()):
        raise ValueError('Teacher initialization provenance differs from compact cache')
    selected = _selected_identities(document['episodes'], samples=plan['normalization']['requested_samples'],
                                    seed=plan['normalization']['seed'])
    identities = [(episode, time) for episode, _, time in selected]
    fit = artifact['fit']
    if (fit['split'] != 'train' or fit['samples'] != len(selected)
            or fit['requested_samples'] != plan['normalization']['requested_samples']
            or fit['seed'] != plan['normalization']['seed']
            or fit['source_index_sha256'] != plan['source_index']['sha256']
            or fit['selected_identities'] != identities or fit['selected_identity_sha256'] != _digest(identities)):
        raise ValueError('Teacher initialization was not fitted to the exact selected train frames')
    teacher.load_state_dict(artifact['teacher_state'], strict=True)
    _check_teacher(teacher, plan, len(selected))
    return dict(fit, index_sha256=document.index_sha256, artifact_sha256=artifact_info['sha256'])


class EpisodeFeatureReader:
    """Bounded CPU LRU; an episode is read once per retained prefix, never per frame."""
    def __init__(self, document, max_episodes=32):
        self.document = document
        self.max_episodes = max_episodes
        if max_episodes < 1:
            raise ValueError('max_episodes must be positive')
        self.cache = OrderedDict()
        self.source_digests = {}

    def raw(self, record):
        name = record['feature_file']
        if name not in self.cache:
            if file_sha256(name) != record['feature_sha256']:
                raise ValueError('Compact feature checksum mismatch')
            payload = torch.load(name, map_location='cpu', weights_only=True)
            self.cache[name] = _check_episode(payload, record, self.document['preparation'])
            self.source_digests[name] = tuple(payload['source_frame_sha256'])
            while len(self.cache) > self.max_episodes:
                evicted, _ = self.cache.popitem(last=False)
                self.source_digests.pop(evicted)
        self.cache.move_to_end(name)
        return self.cache[name]

    def visual_digest(self, record, position):
        if not 0 <= position < len(record['times']):
            raise ValueError('Visual endpoint outside the episode')
        self.raw(record)
        return self.source_digests[record['feature_file']][position]

    def prefix(self, record, end_index, teacher, device='cpu'):
        if not 0 <= end_index < len(record['times']):
            raise ValueError('Prefix endpoint outside the episode')
        raw, valid = self.raw(record)
        teacher_device = teacher.projection.device
        return teacher.normalize_raw(raw[:end_index+1].to(teacher_device),
                                     valid[:end_index+1].to(teacher_device)).to(device)
