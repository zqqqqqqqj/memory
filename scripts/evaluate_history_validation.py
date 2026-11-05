"""One isolated simulator/policy shard for synchronous training validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from qrm.config import ROOT
from qrm.evaluation.manifest import sha256
from qrm.history.validation import atomic_json, validate_shard_results
from qrm.joint.evaluation import resume_evaluation_results, save_evaluation_results


def run(args):
    import torch
    import yaml
    from qrm.joint.collection import simulator_environment
    from qrm.joint.runtime import load_joint_policy
    from qrm.policies.server import policy_server

    out = args.out.resolve()
    plan = json.loads((out/'validation_plan.json').read_text())
    cfg = json.loads((out/'config.json').read_text())
    if not 0 <= args.rank < plan['world_size'] or plan['split'] != 'validation':
        raise ValueError('Invalid validation shard')
    from qrm.history.validation import build_validation_plan
    if build_validation_plan(cfg, {key: plan[key] for key in (
            'checkpoint', 'checkpoint_step', 'checkpoint_sha256', 'checkpoint_size', 'weights', 'method',
            'memory_mode', 'policy_seed', 'pose_encoding', 'norm_stats_sha256', 'source_signature',
            'evaluation_sources')}, plan['world_size']) != plan:
        raise ValueError('Saved validation plan/config mismatch')
    for name, digest in {**plan['source_signature'], **plan['evaluation_sources']}.items():
        source = (ROOT/name).resolve()
        if not source.is_relative_to(ROOT.resolve()) or sha256(source) != digest:
            raise ValueError(f'Validation source changed: {name}')
    checkpoint = Path(plan['checkpoint'])
    if checkpoint.stat().st_size != plan['checkpoint_size'] or sha256(checkpoint) != plan['checkpoint_sha256']:
        raise ValueError('Validation checkpoint changed')
    directory = out/f'rank_{args.rank}'
    directory.mkdir(parents=True, exist_ok=True)
    shard = plan['shards'][args.rank]
    config_text = yaml.safe_dump(cfg)
    if (directory/'evaluation_plan.json').exists():
        results = resume_evaluation_results(directory, shard)
        if (directory/'config.yaml').read_text() != config_text:
            raise ValueError('Validation rollout configuration changed')
    else:
        (directory/'config.yaml').write_text(config_text)
        atomic_json(directory/'evaluation_plan.json', shard)
        results = []
    # An empty shard is legitimate when the number of ranks exceeds jobs.
    if len(results) < len(shard['jobs']):
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible.isdecimal() or os.environ.get('MUJOCO_EGL_DEVICE_ID') != visible:
            raise ValueError('Evaluator and simulator must use the same one physical GPU')
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(0)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if (payload['config'] != cfg or payload['step'] != plan['checkpoint_step']
                or payload['seed'] != plan['policy_seed'] or payload.get('weights') != 'complete_ema'):
            raise ValueError('Loaded validation EMA metadata differs from plan')
        policy, loaded_cfg, loaded_payload = load_joint_policy(checkpoint, device='cuda:0', payload=payload)
        if loaded_cfg != cfg:
            raise ValueError('Loaded policy configuration changed')
        del payload, loaded_payload
        benchmark = Path(cfg['benchmark']['root'])
        with policy_server(policy, port=0) as port:
            for index, ((task, seed), trial) in enumerate(zip(shard['jobs'], shard['trial_indices'], strict=True)):
                if index < len(results):
                    continue
                episode = directory/f'task{task}_seed{seed}'
                if episode.exists() or episode.with_suffix('.log').exists():
                    archive = directory/'previous_attempts'/f'{episode.name}_{time.time_ns()}'
                    archive.mkdir(parents=True)
                    for path in (episode, episode.with_suffix('.log')):
                        if path.exists():
                            path.rename(archive/path.name)
                command = [str(benchmark/'.venv/bin/python'), str(ROOT/'scripts/run_history_benchmark.py'),
                           '--config', str(directory/'config.yaml'), '--tasks', str(task), '--seed', str(seed),
                           '--port', str(port), '--out', str(episode)]
                with episode.with_suffix('.log').open('a') as log:
                    subprocess.run(command, cwd=ROOT, env=simulator_environment(benchmark, 0),
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                records = json.loads((episode/'episodes.json').read_text())
                if not isinstance(records, list) or len(records) != 1:
                    raise ValueError('One simulator episode is required per validation job')
                result = records[0]
                if (result.get('task_id'), result.get('seed')) != (task, seed):
                    raise ValueError('Simulator returned the wrong validation identity')
                result['trial_index'] = trial
                validate_shard_results([result], dict(jobs=[[task, seed]], trial_indices=[trial]))
                results.append(result)
                save_evaluation_results(directory, results)
                print(json.dumps(dict(rank=args.rank, step=plan['checkpoint_step'], task=task, seed=seed,
                                      TSR=result['TSR'], CSR=result['CSR'])), flush=True)
    validate_shard_results(results, shard)
    save_evaluation_results(directory, results)
    atomic_json(directory/'worker_complete.json', dict(rank=args.rank,
        checkpoint_sha256=plan['checkpoint_sha256'], episodes=len(results), status='completed'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rank', type=int, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
