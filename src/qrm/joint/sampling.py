import hashlib
import numpy as np

def select_decision(records, seed, global_index, stride):
    """Uniform task -> episode -> query, independent of rank/world size."""
    digest = hashlib.sha256(f'{seed}:{global_index}'.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    tasks = sorted({r['task_id'] for r in records})
    task = tasks[int(rng.integers(len(tasks)))]
    group = sorted((r for r in records if r['task_id'] == task), key=lambda r: r['episode_id'])
    record = group[int(rng.integers(len(group)))]
    time = int(rng.integers((record['length']+stride-1)//stride))*stride
    return record, time
