"""Prepare episode-contiguous fixed history features and train-only statistics.

This command runs CPU data conversion only. It never starts training or CUDA.
--dry-run prints the planned inputs/outputs without opening visual frame files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from qrm.config import ROOT, load_config
from qrm.history.compact_cache import prepare_features, teacher_spec
from qrm.history.teacher import FixedSpatialTeacher


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT/'data/history/index.json',
                        help='Existing complete frozen visual index (read-only)')
    parser.add_argument('--out', type=Path, default=ROOT/'data/history_fixed')
    parser.add_argument('--config', type=Path, default=ROOT/'ablation_memory_capacity/configs/k8.yaml')
    parser.add_argument('--teacher-fit-frames', type=int, default=16384,
                        help='Deterministic train-only sample count, capped only by available train frames')
    parser.add_argument('--seed', type=int, help='Normalization sample seed; default is history.teacher_seed')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.teacher_fit_frames < 2 or args.cpu_threads < 1:
        parser.error('teacher-fit-frames must be >= 2 and cpu-threads must be positive')
    args.out = args.out.resolve()
    if not args.out.is_relative_to((ROOT/'data').resolve()):
        parser.error('--out must remain inside the new project data directory')
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    h, memory = cfg['history'], cfg['memory']
    kwargs = dict(prefix_dim=2048, image_grid=h['image_grid'], camera_indices=tuple(h['camera_indices']),
                  region_dim=memory['coarse_dim']//(4*len(h['camera_indices'])),
                  seed=h['teacher_seed'], scale_floor=h['teacher_scale_floor'])
    seed = h['teacher_seed'] if args.seed is None else args.seed
    # Dry-run does not create output, scan the large visual index or initialize a model.
    if args.dry_run:
        print(json.dumps(dict(operation='cpu_fixed_feature_preparation', training=False, cuda=False,
                         source_index=str(args.index.resolve()), out=str(args.out), teacher=kwargs,
                         requested_train_frames=args.teacher_fit_frames, normalization_seed=seed,
                         source_index_exists=args.index.is_file(), validation='plan_only'), indent=2))
        return
    torch.set_num_threads(args.cpu_threads)
    teacher = FixedSpatialTeacher(**kwargs).eval().requires_grad_(False)
    document = prepare_features(args.index, args.out, teacher,
                                fit_frames=args.teacher_fit_frames, seed=seed)
    print(json.dumps(dict(status='complete', index=str(args.out/'index.json'),
                         episodes=len(document['episodes']), frames=sum(len(r['times']) for r in document['episodes']),
                         teacher_initialization=document['teacher_initialization'], teacher=teacher_spec(teacher)), indent=2))


if __name__ == '__main__':
    main()
