"""CPU preflight for the pinned dataset/model observation contract.

This intentionally runs before loading a multi-billion-parameter backbone. It
checks source and metadata invariants, not the accuracy of a learned policy.
The offline audit additionally exercises real recorded states and video frames.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from qrm.config import ROOT
from qrm.normalization import validate_norm_stats

CAMERAS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')
STATE_NAMES = ['ee_x', 'ee_y', 'ee_z', 'ee_ax', 'ee_ay', 'ee_az', 'gripper_0', 'gripper_1']
ACTION_NAMES = ['dx', 'dy', 'dz', 'dax', 'day', 'daz', 'gripper']


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def benchmark_prompts(root):
    path = Path(root)/'evaluation_benchmark/scripts/eval_common.py'
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TASK_PROMPTS'
                                               for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Cannot locate the official TASK_PROMPTS mapping')


def validate_input_contract(cfg, index_document=None, *, base_sha256=None):
    """Fail early on incompatible metadata, preprocessing or frozen caches.

    Does not re-hash the large backbone/videos; callers already verify those
    when preparing caches/checkpoints. Pass an independently verified base hash
    to additionally bind an index to that exact checkpoint.
    """
    from .config import validate_config
    validate_config(cfg)
    dataset, model, benchmark = (cfg[k] for k in ('dataset', 'model', 'benchmark'))
    if (model.get('discrete_state_input') is not True or model['action_horizon'] != 20
            or benchmark['replan_steps'] != 10 or benchmark.get('observation_protocol') != 'policy_query_only'):
        raise ValueError('Pinned input contract requires state tokens, horizon 20, execution/observation stride 10')
    if cfg['history']['camera_indices'] != [0, 1] or cfg['history']['image_grid'] != 16:
        raise ValueError('Pinned cameras are main=0, wrist=1, padding=2 with a 16x16 patch grid')
    if dataset.get('use_subtask_labels_as_policy_input') is not False:
        raise ValueError('Privileged subtask labels must not be policy inputs')
    manifest_path = Path(dataset['manifest']).resolve()
    manifest = json.loads(manifest_path.read_text())
    root = Path(dataset['root']).resolve()
    if (manifest.get('schema') != 2 or manifest.get('format') != 'lerobot_v3'
            or Path(manifest['root']).resolve() != root
            or manifest.get('repo_id') != dataset['repo_id']
            or manifest.get('revision') != dataset['revision']):
        raise ValueError('Dataset root/revision/format differs from the bound demonstration manifest')
    norms = validate_norm_stats(dataset['norm_stats'], manifest_path)
    stats = json.loads(Path(dataset['norm_stats']).read_text())['norm_stats']
    for name, width in (('state',8),('actions',7)):
        for key in ('mean','std','q01','q99'):
            values = np.asarray(stats[name][key])
            if values.shape != (width,) or not np.isfinite(values).all():
                raise ValueError(f'Normalization {name}/{key} dimension or values changed')
        if np.any(np.asarray(stats[name]['q99']) < np.asarray(stats[name]['q01'])):
            raise ValueError('Normalization quantiles have an invalid order')
    info_path = root/'meta/info.json'
    entry = next((r for r in manifest['files'] if r['path'] == 'meta/info.json'), None)
    if entry is None or entry['sha256'] != _sha(info_path):
        raise ValueError('Dataset feature metadata changed from the pinned manifest')
    info = json.loads(info_path.read_text())
    features = info['features']
    for name, width, names in (('observation.state', 8, STATE_NAMES), ('action', 7, ACTION_NAMES)):
        feature = features[name]
        if feature['shape'] != [width] or feature['dtype'] != 'float32' or feature['names'] != names:
            raise ValueError(f'Unexpected {name} dimension/order/dtype')
    if info['fps'] != 20 or benchmark['resize_size'] != 256:
        raise ValueError('Pinned videos use 20 Hz control and 256x256 processed observations')
    for camera in ('front', 'wrist'):
        if features[f'observation.images.{camera}']['shape'] != [3, 256, 256]:
            raise ValueError('Dataset camera layout/resolution changed')
    prompts = benchmark_prompts(benchmark['root'])
    identities = set()
    tasks_by_split = {'train': set(), 'validation': set()}
    for record in manifest['episodes']:
        identity = (record['task_id'], record['seed'])
        if identity in identities:
            raise ValueError('Duplicate episode across data partitions')
        identities.add(identity)
        if record['prompt'] != prompts.get(f'task{record["task_id"]}'):
            raise ValueError(f'Train/evaluation task prompt mismatch: {record["episode_id"]}')
        if record['split'] in tasks_by_split:
            tasks_by_split[record['split']].add(record['task_id'])
        if record['fps'] != 20 or set(record['videos']) != {'front', 'wrist'}:
            raise ValueError('Episode camera/frequency metadata changed')
        for camera, video in record['videos'].items():
            if not video['path'].startswith(f'videos/observation.images.{camera}/'):
                raise ValueError('Front/wrist video paths have been exchanged')
            offset = video['from_timestamp'] * 20
            if abs(offset - round(offset)) > .1:
                raise ValueError('Video start does not align with its control frame')
    expected_tasks = set(cfg['training']['tasks'])
    if any(not expected_tasks <= found for found in tasks_by_split.values()):
        raise ValueError('Training or held-out validation tasks are absent')
    openpi = Path(model['openpi_root'])
    commit = subprocess.check_output(['git', '-C', str(openpi), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != model['openpi_commit']:
        raise ValueError('OpenPI revision differs from the pinned input transforms')
    # A matching HEAD alone does not detect local edits to the preprocessing.
    sources = ('src/openpi/policies/libero_policy.py', 'src/openpi/policies/policy_config.py',
               'src/openpi/training/config.py', 'src/openpi/transforms.py',
               'src/openpi/models/tokenizer.py', 'src/openpi/models/model.py',
               'src/openpi/models_pytorch/preprocessing_pytorch.py',
               'src/openpi/models_pytorch/pi0_pytorch.py',
               'src/openpi/models_pytorch/gemma_pytorch.py')
    source_hashes = {}
    for relative in sources:
        actual = (openpi/relative).read_bytes()
        pinned = subprocess.check_output(['git', '-C', str(openpi), 'show', f'{commit}:{relative}'])
        if actual != pinned:
            raise ValueError(f'Unreviewed local change to pinned OpenPI input code: {relative}')
        source_hashes[relative] = hashlib.sha256(actual).hexdigest()
    manifest_hash = _sha(manifest_path)
    if index_document is not None:
        import inspect
        from qrm.joint.model import frozen_visual
        provenance = index_document.get('visual_provenance', {})
        expected = dict(source_manifest_sha256=manifest_hash,
                        norm_stats_sha256=norms['norm_stats_sha256'],
                        preprocess_sha256=_sha(ROOT/'src/qrm/policies/pi05_bridge.py'),
                        encoder_sha256=hashlib.sha256(inspect.getsource(frozen_visual).encode()).hexdigest(),
                        openpi_commit=commit, augmentation=False,
                        encoder='qrm.joint.model.frozen_visual')
        if base_sha256 is not None:
            expected['base_sha256'] = base_sha256
        for key, value in expected.items():
            if provenance.get(key) != value:
                raise ValueError(f'Frozen visual cache provenance mismatch: {key}')
        if index_document.get('stride') != benchmark['replan_steps']:
            raise ValueError('History cache observation grid differs from deployed execution stride')
    return dict(status='passed', manifest_sha256=manifest_hash,
                norm_stats_sha256=norms['norm_stats_sha256'], openpi_commit=commit,
                input_source_sha256=source_hashes, tasks=sorted(expected_tasks),
                state_encoding='signed_xyzw_to_axis_angle_0_to_2pi', state_dimension=8,
                action_dimension=7, prediction_horizon=20, execution_stride=10,
                camera_order=list(CAMERAS), camera_mask=[True, True, False], image_patch_grid=16,
                scope='metadata/source preflight; not a simulator rollout or policy-quality test')


def quaternion_from_recorded_axis_angle(axis_angle):
    """Counterfactual adapter fixture; does NOT claim to recover original raw telemetry."""
    angle_axis = np.asarray(axis_angle, dtype=np.float64)
    if angle_axis.shape != (3,) or not np.isfinite(angle_axis).all():
        raise ValueError('Expected finite recorded axis-angle [3]')
    angle = np.linalg.norm(angle_axis)
    if angle < 1e-12:
        return np.array([0., 0., 0., 1.])
    if angle >= 2*np.pi - 1e-4:
        raise ValueError('Near 2*pi axis-angle is singular under the recorded convention')
    return np.r_[angle_axis/angle*np.sin(angle/2), np.cos(angle/2)]


def cpu_policy_transforms(cfg):
    """Instantiate the actual OpenPI transforms without loading model weights."""
    from dataclasses import replace
    from openpi.training import config
    from openpi import transforms
    from openpi.shared.normalize import deserialize_json
    base = config.get_config('pi05_libero')
    training = replace(base, model=replace(base.model, pytorch_compile_mode=None,
                       discrete_state_input=True, action_horizon=cfg['model']['action_horizon']),
                       data=replace(base.data, repo_id=cfg['dataset']['repo_id']))
    data = training.data.create(training.assets_dirs, training.model)
    if getattr(training.data, 'extra_delta_transform', False):
        raise ValueError('Recorded OSC controls are already deltas')
    stats = deserialize_json(Path(cfg['dataset']['norm_stats']).read_text())
    inputs = transforms.compose([transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
               transforms.Normalize(stats, use_quantiles=data.use_quantile_norm), *data.model_transforms.inputs])
    outputs = transforms.compose([*data.model_transforms.outputs,
               transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm), *data.data_transforms.outputs])
    return inputs, outputs
