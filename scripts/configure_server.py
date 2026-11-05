#!/usr/bin/env python3
"""Write ignored local asset paths without changing experiment configurations."""
import argparse
from pathlib import Path

import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark-root', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--openpi-root', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--norm-stats', type=Path)
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--out', type=Path, default=root/'server.local.yaml')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        parser.error('Local configuration exists; review it or pass --force explicitly')
    assets = args.assets_root.expanduser().resolve()
    benchmark = args.benchmark_root.expanduser().resolve()
    def absolute(value):
        return str(value.expanduser().resolve())
    document = dict(
        dataset=dict(root=absolute(args.dataset_root or assets/'robomemarena_lerobot'),
                     manifest=absolute(args.manifest or assets/'metadata/robomemarena_lerobot_manifest.json'),
                     norm_stats=absolute(args.norm_stats or assets/'metadata/robomemarena_norm/norm_stats.json')),
        benchmark=dict(root=str(benchmark)),
        model=dict(openpi_root=absolute(args.openpi_root or benchmark/'training/openpi'),
                   base_checkpoint_pytorch=absolute(args.base_checkpoint or assets/'pi05_base_pytorch'),
                   base_checkpoint_jax=absolute(assets/'pi05_base')))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('# Local paths only. Do not commit this file.\n'+yaml.safe_dump(document, sort_keys=False))
    print(f'Wrote {args.out.resolve()}')
    print('No data, weights, environments or caches were downloaded or copied.')


if __name__ == '__main__':
    main()
