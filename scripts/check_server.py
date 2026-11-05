#!/usr/bin/env python3
"""Check local dependencies and pinned input contracts without starting training."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

from qrm.config import ROOT, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=ROOT/'data/history_fixed/index.json')
    parser.add_argument('--without-cache', action='store_true')
    args = parser.parse_args()
    cfg = load_config(ROOT/'ablation_memory_capacity/configs/k8.yaml')
    import torch
    torch.set_num_threads(4)
    from qrm.history.input_contract import validate_input_contract
    from qrm.history.compact_cache import read_feature_index, load_teacher_initialization
    from qrm.history.config import build_memory
    for path in (Path(cfg['model']['base_checkpoint_pytorch'])/'model.safetensors',
                 Path(cfg['benchmark']['root'])/'.venv/bin/python',
                 Path(cfg['benchmark']['root'])/'local_setup/libero_config/config.yaml'):
        if not path.is_file():
            raise FileNotFoundError(f'Missing local dependency: {path}')
    index = None if args.without_cache else read_feature_index(args.index)
    validate_input_contract(cfg, index)
    if index is not None:
        load_teacher_initialization(args.index, build_memory(cfg).teacher, document=index)
    import openpi
    import openpi_client
    from websockets.sync.server import serve
    print('Pinned input contract, local asset paths and training imports passed.')
    print('No formal GPU training or simulator rollout was started.')


if __name__ == '__main__':
    main()
