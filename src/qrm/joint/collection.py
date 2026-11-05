"""One isolated schema-4 simulator rollout."""
import os
from pathlib import Path
import json
import subprocess
from qrm.config import ROOT

def simulator_environment(root, device_index):
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'src'),str(root/'evaluation_benchmark/libero_fork'),
                                      str(root/'evaluation_benchmark/scripts'),str(root/'third_party/openpi_minimal/packages/openpi-client/src')])
    env['LIBERO_CONFIG_PATH'] = str(root/'local_setup/libero_config')
    env['MUJOCO_GL'] = env['PYOPENGL_PLATFORM'] = 'egl'
    visible = env.get('CUDA_VISIBLE_DEVICES','').split(',')
    env['MUJOCO_EGL_DEVICE_ID'] = visible[device_index] if visible != [''] else str(device_index)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    return env


def run_simulator(policy, cfg, task, seed, out, port, device_index, config_path):
    root=Path(cfg['benchmark']['root'])
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    # Schema-4 config validation is pure Python, but importing its package
    # initializer would import torch in the deliberately lightweight simulator.
    entry = 'run_history_benchmark.py' if cfg.get('schema_version') == 4 else 'run_benchmark.py'
    cmd=[str(root/'.venv/bin/python'),str(ROOT/'scripts'/entry),'--config',str(config_path),
         '--tasks',str(task),'--seed',str(seed),'--port',str(port),'--out',str(out)]
    with out.with_suffix('.log').open('a') as log:
        subprocess.run(cmd,cwd=ROOT,env=simulator_environment(root,device_index),stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((out/'episodes.json').read_text())[0]
    if result['status']!='completed':
        raise RuntimeError(result)
    return result
