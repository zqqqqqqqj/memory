import json
import math
import os
from pathlib import Path
import subprocess
import sys
from qrm.config import ROOT


def resume_evaluation_results(out, plan):
    """Resume only a validated prefix of the exact same evaluation plan."""
    out = Path(out)
    saved = json.loads((out/'evaluation_plan.json').read_text())
    if saved != json.loads(json.dumps(plan)):
        raise ValueError('Resume evaluation plan/checkpoint mismatch')
    path = out/'episodes.json'
    results = json.loads(path.read_text()) if path.exists() else []
    if len(results) > len(plan['jobs']):
        raise ValueError('More saved episodes than planned jobs')
    for index, result in enumerate(results):
        task, seed = plan['jobs'][index]
        if (result.get('task_id'), result.get('seed'), result.get('trial_index')) != (
                task, seed, plan['trial_indices'][index]):
            raise ValueError('Saved episodes are not an ordered prefix of the evaluation plan')
        if result.get('status') != 'completed' or result.get('errors'):
            raise ValueError('Cannot skip an incomplete/error episode when resuming')
        if any(not math.isfinite(result[k]) or not 0 <= result[k] <= 1 for k in ('TSR', 'CSR')):
            raise ValueError('Invalid saved evaluation score')
    return results


def save_evaluation_results(out, results):
    """Keep the last complete aggregate if interrupted during JSON writing."""
    out = Path(out)
    temporary = out/'episodes.json.tmp'
    with temporary.open('w') as stream:
        json.dump(results, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(out/'episodes.json')


def plan_evaluation(cfg, *, split='validation', tasks=None, seed_start=None,
                    fixed_seed=None, episodes_per_task=None, shard_index=0, shard_count=1):
    if split not in {'validation', 'test'}:
        raise ValueError('Unknown evaluation split')
    tasks = list(cfg['training']['tasks'] if tasks is None else tasks)
    if not tasks or len(tasks) != len(set(tasks)) or not set(tasks) <= set(cfg['training']['tasks']):
        raise ValueError('Evaluation requires distinct tasks covered by this checkpoint')
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('Require shard-count >= 1 and 0 <= shard-index < shard-count')
    low, high = cfg['benchmark'][split+'_seeds']
    if fixed_seed is not None:
        if seed_start is not None:
            raise ValueError('Choose fixed-seed or seed-start, not both')
        count = 1 if episodes_per_task is None else episodes_per_task
        if count < 1 or not low <= fixed_seed <= high:
            raise ValueError(f'Fixed seed must be within {split} range {low}..{high}; count must be positive')
        seeds = [fixed_seed]*count
    else:
        start = low if seed_start is None else seed_start
        count = high-start+1 if episodes_per_task is None else episodes_per_task
        if count < 1 or start < low or start+count-1 > high:
            raise ValueError(f'Evaluation seeds must be distinct and within {split} range {low}..{high}')
        seeds = list(range(start, start+count))
    jobs = [(task, seed) for task in tasks for seed in seeds]
    selected = jobs[shard_index::shard_count]
    return dict(split=split, tasks=tasks, seeds=seeds, episodes_per_task=count,
                seed_mode='fixed' if fixed_seed is not None else 'consecutive',
                fixed_seed=fixed_seed, unique_reset_seeds_per_task=len(set(seeds)),
                registered_seed_range=[low, high], total_episodes=len(jobs),
                shard_index=shard_index, shard_count=shard_count,
                shard_episodes=len(selected), jobs=selected,
                trial_indices=[i % count for i in range(len(jobs))][shard_index::shard_count])


def aggregate(results,tasks):
    per_task={}
    for task in tasks:
        values=[r for r in results if r['task_id']==task]
        if not values or any(r['status']!='completed' for r in values):
            raise ValueError('Missing/failed validation episodes')
        per_task[str(task)]={k:sum(float(r[k]) for r in values)/len(values) for k in ('TSR','CSR')}
        per_task[str(task)]['episodes']=len(values)
    return dict(per_task=per_task,macro_TSR=sum(r['TSR'] for r in per_task.values())/len(per_task),
                macro_CSR=sum(r['CSR'] for r in per_task.values())/len(per_task),
                evaluation_env_steps=sum(r['native_env_steps'] for r in results))


def evaluate_distributed(checkpoint,cfg,ctx,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(ROOT/'scripts/evaluate_joint.py'),'--checkpoint',str(checkpoint),
             '--device',str(ctx.device),'--shard-index',str(ctx.rank),'--shard-count',str(ctx.world_size),
             '--out',str(out/f'rank_{ctx.rank}'),'--tasks',*map(str,cfg['training']['tasks'])]
    env=dict(os.environ)
    for name in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(name,None)
    with (out/f'rank_{ctx.rank}.log').open('a') as log:
        subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    values=json.loads((out/f'rank_{ctx.rank}'/'episodes.json').read_text())
    gathered=ctx.gather(values)
    if not ctx.primary:return None
    result=aggregate([v for group in gathered for v in group],cfg['training']['tasks'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
