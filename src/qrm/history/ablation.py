"""Table-2 experiment entry points; preview by default, execute explicitly."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from qrm.config import ROOT, load_config


def experiment_output(experiment, seed, stage):
    return ROOT/'runs'/'ablations'/experiment/f'seed_{seed}'/stage


def launch_arguments(suite, args):
    """Resolve a table row, including the common reference memory checkpoint."""
    suite = Path(suite).resolve()
    experiments = json.loads((suite/'experiments.json').read_text())['variants']
    if args.variant not in experiments:
        raise ValueError(f'Unknown variant {args.variant!r}; choose from {", ".join(experiments)}')
    row = experiments[args.variant]
    config = suite/row['config']
    cfg = load_config(config)
    source = row.get('memory_source', row['experiment'])
    pretrained = cfg['sft'].get('memory_initialization', 'pretrained') == 'pretrained'
    if args.stage == 'memory':
        if not pretrained:
            raise ValueError('This variant skips memory pretraining; launch --stage action directly')
        if args.memory_checkpoint:
            raise ValueError('--memory-checkpoint is an action-stage option')
        if source == 'reference':
            config = ROOT/'ablation_memory_capacity/configs/k8.yaml'
    out = args.out or experiment_output(source if args.stage == 'memory' else row['experiment'],
                                          args.seed, args.stage)
    result = ['--stage', args.stage, '--config', str(config), '--gpus', args.gpus,
              '--out', str(out), '--index', str(args.index), '--seed', str(args.seed)]
    if args.stage == 'action':
        if not pretrained and args.memory_checkpoint:
            raise ValueError('This variant uses fresh memory and rejects --memory-checkpoint')
        if pretrained:
            step = cfg['stages']['memory']['action_checkpoint_step']
            checkpoint = args.memory_checkpoint or experiment_output(source, args.seed, 'memory')/f'step_{step:06d}.pt'
            result += ['--memory-checkpoint', str(checkpoint)]
    if args.steps is not None:
        result += ['--steps', str(args.steps)]
    if args.resume:
        result.append('--resume')
    if not args.execute:
        result.append('--dry-run')
    return result


def main(suite):
    suite = Path(suite).resolve()
    description = json.loads((suite/'experiments.json').read_text())
    parser = argparse.ArgumentParser(description=description['description'])
    parser.add_argument('--list', action='store_true', help='List variants without loading data or GPUs')
    parser.add_argument('--variant', choices=list(description['variants']))
    parser.add_argument('--stage', choices=('memory', 'action'))
    parser.add_argument('--gpus', help='Physical GPU ids: one/two/four for memory, two/four for action')
    parser.add_argument('--index', type=Path, default=ROOT/'data/history_fixed/index.json')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--memory-checkpoint', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resume', action='store_true')
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--execute', action='store_true', help='Start training; otherwise preview only')
    execution.add_argument('--dry-run', action='store_true', help='Explicit preview (the default)')
    args = parser.parse_args()
    if args.list:
        for name, row in description['variants'].items():
            cfg = load_config(suite/row['config'])
            print(f"{name:28s} K={cfg['memory']['slots']:2d} "
                  f"residual={cfg['memory']['residual_dim'] if cfg['memory'].get('residual_enabled', True) else 0:3d} "
                  f"init={cfg['sft'].get('memory_initialization', 'pretrained'):10s} "
                  f"joint={cfg['history'].get('joint_supervision', True)}  {row['label']}")
        return
    if args.variant is None or args.stage is None or args.gpus is None:
        parser.error('Specify --variant, --stage and --gpus, or use --list')
    try:
        command = launch_arguments(suite, args)
    except ValueError as error:
        parser.error(str(error))
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT/'src'), environment.get('PYTHONPATH'))))
    subprocess.run([sys.executable, '-m', 'qrm.history.launcher', *command], cwd=ROOT,
                   env=environment, check=True)
