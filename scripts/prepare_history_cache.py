"""Export complete train/validation visual histories; reuse old cache read-only.

No GPU is used unless --encode-missing is explicitly supplied. Every omitted
frame is reported as an error; an incomplete history index is never emitted.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import timedelta
import hashlib
import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from qrm.config import ROOT, load_config
from qrm.data import decisions, load_manifest
from qrm.history.data import cache_key, file_sha256, validate_index
from qrm.history.input_contract import validate_input_contract
from qrm.joint.model import VisualCache, frozen_visual
from qrm.normalization import validate_norm_stats
from qrm.policies.pi05_bridge import Pi05Bridge


def atomic_json(path, document):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(document, indent=2) + '\n')
    temp.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'ablation_memory_capacity/configs/k8.yaml')
    parser.add_argument('--out', type=Path, default=ROOT/'data/history')
    parser.add_argument('--reuse-project', type=Path,
                        help='Original project root; verify its old cache namespace and only read its cache files')
    parser.add_argument('--encode-missing', action='store_true', help='Run frozen pi05 vision for every missing frame')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--tasks', type=int, nargs='+', help='Explicit task subset, applied equally to train/validation')
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not args.out.is_relative_to((ROOT/'data').resolve()):
        raise ValueError('--out must be inside this new project data directory; original caches stay read-only')
    return args


def prepare_export(args):
    """Rank zero alone verifies provenance and scans existing caches, entirely on CPU."""
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    input_contract = validate_input_contract(cfg)
    atomic_json(args.out/'input_contract.json', input_contract)
    source_manifest = Path(cfg['dataset']['manifest']).resolve()
    actual_commit = subprocess.check_output(['git', '-C', cfg['model']['openpi_root'], 'rev-parse', 'HEAD'],
                                            text=True).strip()
    if actual_commit != cfg['model']['openpi_commit']:
        raise ValueError('OpenPI source revision differs from the pinned visual preprocessing provenance')
    print('Verifying dataset, training-only normalization and frozen backbone hashes...', flush=True)
    records = load_manifest(source_manifest, 'train') + load_manifest(source_manifest, 'validation')
    if args.tasks:
        records = [record for record in records if record['task_id'] in args.tasks]
        if {r['task_id'] for r in records} != set(args.tasks):
            raise ValueError('Requested task missing from demonstration manifest')
    norms = validate_norm_stats(cfg['dataset']['norm_stats'], source_manifest)
    base = Path(cfg['model']['base_checkpoint_pytorch']).resolve()
    base_hash = file_sha256(base/'model.safetensors')
    source_hash = file_sha256(source_manifest)
    bridge_hash = file_sha256(ROOT/'src/qrm/policies/pi05_bridge.py')
    stride = cfg['benchmark']['replan_steps']
    provenance = dict(base_sha256=base_hash, norm_stats_sha256=norms['norm_stats_sha256'],
                      source_manifest_sha256=source_hash, preprocess_sha256=bridge_hash,
                      encoder_sha256=hashlib.sha256(inspect.getsource(frozen_visual).encode()).hexdigest(),
                      openpi_commit=cfg['model'].get('openpi_commit'),
                      encoder='qrm.joint.model.frozen_visual', augmentation=False)
    version = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    provenance['cache_version'] = version
    cache = VisualCache(args.out/'visual_cache', version)
    reuse_root = None
    reuse_info = None
    if args.reuse_project:
        old = args.reuse_project.resolve()
        old_cfg = yaml.safe_load((old/'configs/robomemarena.yaml').read_text())
        if (Path(old_cfg['model']['base_checkpoint_pytorch']).resolve() != base
                or file_sha256(old_cfg['dataset']['manifest']) != source_hash
                or file_sha256(old_cfg['dataset']['norm_stats']) != norms['norm_stats_sha256']
                or file_sha256(old/'src/qrm/policies/pi05_bridge.py') != bridge_hash):
            raise ValueError('Old cache backbone/manifest/normalization/preprocessing differs; omit --reuse-project')
        old_model_hash = file_sha256(old/'src/qrm/joint/model.py')
        old_version = hashlib.sha256((base_hash + source_hash + old_model_hash + bridge_hash
                                      + norms['norm_stats_sha256']).encode()).hexdigest()
        reuse_root = old/'data/visual_cache_20261012'/old_version
        if not reuse_root.is_dir():
            raise ValueError(f'Cannot verify original cache namespace {reuse_root}. '
                             'Omit --reuse-project and encode a fresh cache; do not rename old cache directories.')
        # Verify the encoder itself, independently of the whole-file old namespace.
        import ast
        old_text = (old/'src/qrm/joint/model.py').read_text()
        old_tree = ast.parse(old_text)
        old_func = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'frozen_visual')
        new_func = ast.parse(inspect.getsource(frozen_visual)).body[0]
        if ast.dump(old_func, include_attributes=False) != ast.dump(new_func, include_attributes=False):
            raise ValueError('Original frozen vision encoder differs; reuse is unsafe')
        reuse_info = dict(root=str(reuse_root), namespace=old_version,
                          old_model_sha256=old_model_hash, mode='read_only')
    exported, missing = [], []
    reused = present = 0
    for record in records:
        times = list(range(0, record['length'], stride))
        paths = []
        absent = []
        for time in times:
            key = cache_key(record['episode_id'], time)
            own = cache.root/key
            old_path = reuse_root/key if reuse_root else None
            if own.is_file():
                path = own; present += 1
            elif old_path is not None and old_path.is_file():
                path = old_path; reused += 1
            else:
                path = own; absent.append(time)
            paths.append(str(path))
        exported.append(dict(episode_id=record['episode_id'], split=record['split'],
                             task_id=record['task_id'], seed=record['seed'], length=record['length'],
                             source_record=record, times=times, cache_files=paths))
        if absent:
            missing.append((record, absent))
    report = dict(status='missing_frames' if missing else 'complete', total_episodes=len(records),
                  total_frames=sum(len(r['times']) for r in exported), reused_read_only_frames=reused,
                  existing_new_frames=present, missing_frames=sum(len(ts) for _, ts in missing),
                  missing_by_episode=[dict(episode_id=r['episode_id'], split=r['split'], frames=len(ts))
                                      for r, ts in missing], visual_provenance=provenance, reuse=reuse_info)
    atomic_json(args.out/'cache_status.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('missing_by_episode','visual_provenance','reuse')}), flush=True)
    document = dict(schema=1, kind='history_visual_index', stride=stride,
                    source_manifest=dict(path=str(source_manifest), sha256=source_hash),
                    visual_provenance=provenance, reuse=reuse_info,
                    cache_roots=[str(cache.root)] + ([str(reuse_root)] if reuse_root else []), episodes=exported)
    validate_index(document, check_files=False)
    return dict(cfg=cfg, norms=norms['norm_stats'], base=str(base), version=version,
                missing=missing, report=report, document=document)


def partition_missing(missing, world_size, stride):
    """Assign each complete missing episode to exactly one rank, balancing frame counts."""
    if world_size < 1:
        raise ValueError('world_size must be positive')
    seen = set()
    for record, times in missing:
        if record['episode_id'] in seen:
            raise ValueError('Duplicate missing episode')
        seen.add(record['episode_id'])
        legal = set(range(0, record['length'], stride))
        if not times or times != sorted(set(times)) or not set(times).issubset(legal):
            raise ValueError('Missing times must be an ordered unique subset of the episode decision grid')
    shards, loads = [[] for _ in range(world_size)], [0] * world_size
    # Stable tie breaking keeps ownership reproducible, including on resume.
    for item in sorted(missing, key=lambda item: -len(item[1])):
        rank = min(range(world_size), key=lambda index: (loads[index], index))
        shards[rank].append(item)
        loads[rank] += len(item[1])
    return shards


def encode_shard(shard, *, bridge, cache, stride, rank):
    encoded = 0
    for episode_number, (record, times) in enumerate(shard, 1):
        needed, observed = set(times), set()
        if record.get('format') == 'lerobot_v3':
            from qrm.lerobot_v3 import episode_decisions
            stream = episode_decisions(record, bridge.horizon, stride, timesteps=times)
        else:
            stream = decisions(record, bridge.horizon, stride)
        with torch.no_grad(), closing(stream):
            for time, raw, _actions in stream:
                if time not in needed:
                    continue
                if time in observed:
                    raise ValueError(f'Duplicate decision {record["episode_id"]}/{time}')
                observation, _ = bridge.transform(raw)
                prep = bridge.model._preprocess_observation(observation, train=False)
                cache.get((record['episode_id'], time), lambda: frozen_visual(bridge, prep), 'cpu')
                observed.add(time)
        if observed != needed:
            raise ValueError(f'Decision stream omitted requested frames for {record["episode_id"]}: '
                             f'{sorted(needed - observed)}')
        encoded += len(observed)
        print(f'[rank {rank}] Encoded episode {episode_number}/{len(shard)}: '
              f'{record["episode_id"]}, {len(times)} frames; rank total {encoded}', flush=True)
    return encoded


def publish_export(out, plan, statuses):
    """Only the coordinator publishes, after every worker and every path is verified."""
    failures = [f'rank {rank}: {status["error"]}' for rank, status in enumerate(statuses)
                if status.get('error')]
    if failures:
        raise RuntimeError('Frozen-cache encoding failed; no index was published. ' + '; '.join(failures))
    expected = plan['report']['missing_frames']
    if sum(status['encoded'] for status in statuses) != expected:
        raise ValueError('Encoded frame count differs from the complete missing-frame plan; no index was published')
    document, report = plan['document'], plan['report']
    validate_index(document)
    atomic_json(out/'index.json', document)
    report.update(status='complete', encoded_frames=expected, missing_frames=0,
                  encoding_world_size=len(statuses), encoded_frames_by_rank=[s['encoded'] for s in statuses])
    atomic_json(out/'cache_status.json', report)
    print(f'Complete causal train/validation index: {out/"index.json"}', flush=True)


def main():
    args = parse_args()
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    if distributed:
        # CPU metadata coordination: scanning and hashing must not allocate any CUDA memory.
        dist.init_process_group('gloo', timeout=timedelta(hours=6))
    rank = dist.get_rank() if distributed else 0
    try:
        envelope = [None]
        if rank == 0:
            try:
                envelope[0] = dict(plan=prepare_export(args))
            except Exception as exc:
                envelope[0] = dict(error=f'{type(exc).__name__}: {exc}')
        if distributed:
            dist.broadcast_object_list(envelope, src=0)
        if 'error' in envelope[0]:
            raise RuntimeError(envelope[0]['error'])
        plan = envelope[0]['plan']
        if plan['missing'] and not args.encode_missing:
            raise SystemExit('Complete history export blocked by missing frozen frames. Re-run this command '
                             'with --encode-missing (and retain --reuse-project if used). '
                             'For four GPUs, use torchrun --standalone --nproc_per_node=4. '
                             'No incomplete index was written. See cache_status.json.')
        shards = partition_missing(plan['missing'], world_size, plan['document']['stride'])
        if rank == 0:
            print('Missing frames by rank: ' + str([sum(len(ts) for _, ts in shard) for shard in shards]), flush=True)
        status = dict(encoded=0, error=None)
        try:
            if shards[rank]:
                device = args.device
                if distributed and device.startswith('cuda'):
                    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
                if device.startswith('cuda'):
                    torch.cuda.set_device(device)
                bridge = Pi05Bridge(Path(plan['base']), device=device, norm_stats=plan['norms'],
                                    action_horizon=plan['cfg']['model']['action_horizon'])
                bridge.model.eval()
                bridge.model.requires_grad_(False)
                cache = VisualCache(args.out/'visual_cache', plan['version'])
                status['encoded'] = encode_shard(shards[rank], bridge=bridge, cache=cache,
                                                 stride=plan['document']['stride'], rank=rank)
        except Exception as exc:
            status['error'] = f'{type(exc).__name__}: {exc}'
        statuses = [None] * world_size
        if distributed:
            dist.all_gather_object(statuses, status)
        else:
            statuses[0] = status
        completed = [None]
        if rank == 0:
            try:
                publish_export(args.out, plan, statuses)
            except Exception as exc:
                completed[0] = f'{type(exc).__name__}: {exc}'
        if distributed:
            dist.broadcast_object_list(completed, src=0)
        if completed[0]:
            raise RuntimeError(completed[0])
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
