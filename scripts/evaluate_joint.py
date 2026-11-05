import argparse
import json
from pathlib import Path
import time
import torch
import yaml
from qrm.joint.runtime import load_joint_policy
from qrm.joint.collection import run_simulator
from qrm.joint.evaluation import (aggregate, plan_evaluation, resume_evaluation_results,
                                  save_evaluation_results)
from qrm.policies.server import policy_server


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--benchmark-root',type=Path,
                   help='Use an existing simulator installation when evaluating a relocated checkpoint')
    p.add_argument('--tasks',nargs='+',type=int)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    seed_options=p.add_mutually_exclusive_group()
    seed_options.add_argument('--seed-start',type=int,help='First environment reset seed in the selected split')
    seed_options.add_argument('--fixed-seed',type=int,help='Use this environment reset seed for every trial')
    p.add_argument('--episodes-per-task',type=int,help='Trials per task; consecutive seeds by default, repeated seed with --fixed-seed')
    p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--shard-count',type=int,default=1)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',action='store_true',help='Continue the same plan in an existing output directory')
    a=p.parse_args()
    if a.episodes_per_task is not None and a.episodes_per_task < 1:
        p.error('--episodes-per-task must be positive')
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        p.error('Require shard-count >= 1 and 0 <= shard-index < shard-count')
    if a.resume:
        if not (a.out/'evaluation_plan.json').is_file():
            p.error('--resume requires an existing evaluation_plan.json')
    else:
        if a.out.exists():raise FileExistsError('Use a new evaluation directory or --resume')
        a.out.mkdir(parents=True)
    policy,cfg,payload=load_joint_policy(a.checkpoint,device=a.device)
    if a.benchmark_root is not None:
        import copy
        cfg=copy.deepcopy(cfg)
        cfg['benchmark']['root']=str(a.benchmark_root.resolve(strict=True))
    config=a.out/'config.yaml'
    plan=plan_evaluation(cfg,split=a.split,tasks=a.tasks,seed_start=a.seed_start,
                         fixed_seed=a.fixed_seed,
                         episodes_per_task=a.episodes_per_task,
                         shard_index=a.shard_index,shard_count=a.shard_count)
    plan.update(checkpoint=str(Path(a.checkpoint).resolve()), checkpoint_step=payload.get('step'),
                policy_seed_base=payload['seed'])
    del payload
    results=resume_evaluation_results(a.out,plan) if a.resume else []
    if not a.resume:
        config.write_text(yaml.safe_dump(cfg))
        (a.out/'evaluation_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    elif config.read_text() != yaml.safe_dump(cfg):
        raise ValueError('Saved evaluation config differs from checkpoint config')
    jobs=plan['jobs']
    print(f"Evaluation: split={a.split}; tasks={len(plan['tasks'])}; "
          f"seed mode={plan['seed_mode']}; "
          f"reset seeds={plan['seeds'][0]}..{plan['seeds'][-1]}; "
          f"episodes={plan['total_episodes']}; shard episodes={len(jobs)}",flush=True)
    if a.resume:
        with (a.out/'resume_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(completed=len(results),remaining=len(jobs)-len(results),
                                         resumed_unix=time.time(),checkpoint=plan['checkpoint']))+'\n')
        print(f'Resuming evaluation: {len(results)}/{len(jobs)} complete; remaining={len(jobs)-len(results)}',flush=True)
    completed=len(results)
    with policy_server(policy,port=0) as port:
        for index,((task,seed),trial) in enumerate(zip(jobs,plan['trial_indices'],strict=True),1):
            if index <= completed:
                continue
            name=f'task{task}_seed{seed}'
            if plan['seed_mode'] == 'fixed':
                name+=f'_trial{trial+1:03d}'
            if a.resume:
                failed=a.out/name
                if failed.exists() or failed.with_suffix('.log').exists():
                    archive=a.out/'previous_attempts'/f'{name}_{time.time_ns()}'
                    archive.mkdir(parents=True)
                    if failed.exists():failed.rename(archive/name)
                    log=failed.with_suffix('.log')
                    if log.exists():log.rename(archive/log.name)
            result=run_simulator(policy,cfg,task,seed,a.out/name,port,
                                 torch.device(a.device).index or 0,config.resolve())
            result['task_id']=task
            result['trial_index']=trial
            results.append(result)
            save_evaluation_results(a.out,results)
            print(f"[evaluation {index}/{len(jobs)}] task={task} seed={seed} "
                  f"trial={trial+1}/{plan['episodes_per_task']} "
                  f"TSR={result['TSR']} CSR={result['CSR']}",flush=True)
    if not jobs:save_evaluation_results(a.out,[])
    if a.shard_count == 1:
        metrics=aggregate(results,plan['tasks'])
        metrics.update(split=a.split,episodes_per_task=plan['episodes_per_task'],
                       total_episodes=len(results),seeds=plan['seeds'],seed_mode=plan['seed_mode'],
                       unique_reset_seeds_per_task=plan['unique_reset_seeds_per_task'])
        (a.out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')


if __name__=='__main__':main()
