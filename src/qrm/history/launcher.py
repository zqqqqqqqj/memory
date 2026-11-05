"""Validated one/two/four-GPU memory and two/four-GPU action launches; dry-run is read-only."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import csv
import fcntl
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

from qrm.config import ROOT, load_config

LEGACY_ROOT = ROOT/'.no_legacy_project'


def gpu_ids(value):
    pieces = value.split(',')
    if len(pieces) not in (1, 2, 4) or any(not item.isdecimal() for item in pieces):
        raise argparse.ArgumentTypeError('--gpus requires one, two or four physical GPU numbers, e.g. 4, 4,5 or 4,5,6,7')
    ids = [int(item) for item in pieces]
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError('GPU numbers must be distinct')
    return ids


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--stage', required=True, choices=('memory', 'action'))
    result.add_argument('--gpus', type=gpu_ids,
                        help='Default: memory=4, action=4,5,6,7; action requires two or four GPUs')
    result.add_argument('--config', type=Path, default=ROOT/'ablation_memory_capacity/configs/k8.yaml')
    result.add_argument('--index', type=Path, default=ROOT/'data/history_fixed/index.json')
    result.add_argument('--out', type=Path)
    result.add_argument('--memory-checkpoint', type=Path)
    result.add_argument('--steps', type=int)
    result.add_argument('--seed', type=int)
    result.add_argument('--batch-size', type=int, help='Global memory-pretraining batch, divisible by GPU count')
    result.add_argument('--tasks', type=int, nargs='+', help='Action stage only; match exported cache tasks')
    result.add_argument('--reader-mode', choices=('full', 'coarse', 'time'), default=None)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--dry-run', action='store_true', help='Check artifacts and print launch; no GPU access or output writes')
    return result


def required_file(value, purpose):
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(f'Missing {purpose}: {path}')
    return path


def build_launch(args, *, root=ROOT):
    """Pure preparation: inspect inputs without locks, CUDA, or output creation."""
    root = Path(root).resolve()
    if args.gpus is None:
        args.gpus = [4] if args.stage == 'memory' else [4, 5, 6, 7]
    world_size = len(args.gpus)
    if args.stage == 'action' and world_size not in (2, 4):
        raise ValueError('Action training requires two or four GPUs')
    config = required_file(args.config, 'configuration')
    cfg = load_config(config)
    if cfg.get('schema_version') != 4:
        raise ValueError('History launcher requires schema_version=4')
    memory_mode = cfg['sft'].get('memory_mode', 'full')
    default_name = (f'history_memory_{world_size}gpu' if args.stage == 'memory' else
                    f'history_action_baseline_{world_size}gpu' if memory_mode == 'none' else
                    f'history_action_full_{world_size}gpu')
    out = (args.out or root/'runs'/default_name).resolve()
    if not out.is_relative_to((root/'runs').resolve()) or out == (root/'runs').resolve():
        raise ValueError('--out must be a stage directory beneath this new project runs/')
    if out.exists() and not args.resume:
        raise ValueError(f'Output already exists: {out}; use --resume for this exact run or a fresh --out')
    if args.resume and not out.is_dir():
        raise ValueError(f'Cannot resume a missing output directory: {out}')
    if args.steps is not None and args.steps < 1:
        raise ValueError('--steps must be positive')
    index = required_file(args.index, 'complete fixed-feature index; run scripts/prepare_history_features.py first')
    command = [sys.executable, '-u']
    if world_size > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                    f'--nproc_per_node={world_size}']
    if args.stage == 'memory':
        if args.tasks or args.memory_checkpoint:
            raise ValueError('Memory pretraining rejects action-only task/checkpoint flags')
        if memory_mode != 'full':
            raise ValueError('Memory pretraining requires the full-memory configuration, not the no-memory baseline config')
        if not cfg['memory'].get('residual_enabled', True):
            raise ValueError('Coarse-only ablation has no memory-pretraining stage')
        if cfg['sft'].get('memory_initialization', 'pretrained') == 'random':
            raise ValueError('Random-initialized ablation skips memory pretraining')
        batch = args.batch_size if args.batch_size is not None else cfg['stages']['memory']['global_batch_size']
        if batch < 1 or batch % world_size:
            raise ValueError('Pretraining global --batch-size must be positive and divisible by GPU count')
        command += [str(root/'scripts/train_history_memory.py'), '--index', str(index),
                    '--batch-size', str(batch), '--reader-mode',
                    args.reader_mode or cfg['history'].get('reader_mode', 'full')]
        if args.resume:
            required_file(out/'latest.pt', 'pretraining resume checkpoint')
    else:
        if args.batch_size is not None or args.reader_mode not in {None, 'full'}:
            raise ValueError('Action stage takes global batch from config; pretraining-only options are invalid')
        from qrm.history.config import resolve_stage
        from qrm.joint.config import accumulation_steps
        resolved = resolve_stage(cfg, args.stage, args.tasks)
        accumulation_steps(resolved, world_size)
        command += [str(root/'scripts/train_history_sft.py'), '--history-index', str(index)]
        if memory_mode == 'none' or resolved['sft'].get('memory_initialization', 'pretrained') == 'random':
            if args.memory_checkpoint:
                raise ValueError('The no-memory or random-memory experiment starts from official base weights and rejects --memory-checkpoint')
        else:
            memory = required_file(args.memory_checkpoint or root/'runs/history_memory_1gpu/latest.pt',
                                   'memory checkpoint; complete memory stage first')
            command += ['--memory-checkpoint', str(memory)]
        if args.tasks:
            command += ['--tasks', *map(str, args.tasks)]
        if args.resume:
            from qrm.joint.checkpoint import latest_complete_checkpoint
            latest_complete_checkpoint(out)
    command += ['--config', str(config), '--stage', args.stage, '--out', str(out), '--device', 'cuda:0']
    if args.steps is not None:
        command += ['--steps', str(args.steps)]
    if args.seed is not None:
        command += ['--seed', str(args.seed)]
    if args.resume:
        command.append('--resume')
    return command, out


@contextmanager
def gpu_locks(gpus, *, root=ROOT, legacy_root=LEGACY_ROOT):
    """Hold both projects' matching GPU locks; legacy files are never created/written."""
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            for project in (Path(root), Path(legacy_root)):
                path = project/'runs'/f'pi05_sft_gpu{gpu}.lock'
                if project == Path(legacy_root):
                    if not path.exists():
                        continue
                    stream = stack.enter_context(path.open('rb'))
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    stream = stack.enter_context(path.open('a+b'))
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError(f'GPU {gpu} is held by another training launcher: {path}') from error
        yield


def ensure_idle_gpus(gpus, *, run=subprocess.run):
    """Refuse to start over an existing compute process, including other projects."""
    def query(argument):
        result = run(['nvidia-smi', argument, '--format=csv,noheader,nounits'],
                     check=True, text=True, capture_output=True)
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    devices = {int(index): uuid.strip() for index, uuid in query('--query-gpu=index,uuid')}
    if missing := set(gpus)-devices.keys():
        raise ValueError(f'Physical GPUs do not exist: {sorted(missing)}')
    selected = {devices[gpu]: gpu for gpu in gpus}
    busy = [(selected[uuid], pid, name) for uuid, pid, name in
            query('--query-compute-apps=gpu_uuid,pid,process_name') if uuid in selected]
    if busy:
        raise ValueError(f'Selected GPUs already have compute processes: {busy}; choose idle GPUs')


def training_environment(gpus):
    """Use the project's verified W&B connection without exposing credentials."""
    mode = os.environ.get('WANDB_MODE', 'disabled')
    if mode == 'online':
        connection = runpy.run_path(str(ROOT/'scripts/wandb_connection.py'))
        # This only changes this launcher's environment. Existing credentials
        # remain in their original protected file and never enter command/log text.
        receipt = connection['check_connection']()
        print('W&B connection: '+json.dumps(receipt), flush=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       JAX_PLATFORMS='cpu', WANDB_MODE=mode, PYTHONDONTWRITEBYTECODE='1')
    # A fresh launch must not inherit a calling torchrun rank's rendezvous.
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                'ROLE_RANK', 'ROLE_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        environment.pop(key, None)
    return environment


def main(argv=None):
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    try:
        command, out = build_launch(args)
        visible = ','.join(map(str, args.gpus))
        print(f'CUDA_VISIBLE_DEVICES={visible} {shlex.join(command)}', flush=True)
        if args.dry_run:
            print('Dry run: required paths checked; no GPU access, output writes, or training. '
                  'Checkpoint/cache contents and provenance are checked by the training entrypoint.', flush=True)
            return 0
        with gpu_locks(args.gpus):
            ensure_idle_gpus(args.gpus)
            environment = training_environment(args.gpus)
            # The trainer owns creation of out itself and rejects accidental reuse.
            out.parent.mkdir(parents=True, exist_ok=True)
            log_path = out.parent/(out.name+'.console.log')
            with log_path.open('a') as log:
                log.write(json.dumps(dict(command=command, gpus=args.gpus, output=str(out)))+'\n')
                log.flush()
                with subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
                    try:
                        for line in child.stdout:
                            print(line, end='', flush=True)
                            log.write(line)
                            log.flush()
                        return child.wait()
                    except BaseException:
                        child.terminate()
                        child.wait()
                        raise
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        argument_parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
