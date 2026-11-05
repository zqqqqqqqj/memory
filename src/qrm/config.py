"""Schema-4 ablation configurations with optional server-local paths."""
from pathlib import Path

import yaml

from qrm.paths import apply_server_paths

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = path or ROOT/'ablation_memory_capacity/configs/k8.yaml'
    cfg = apply_server_paths(yaml.safe_load(Path(path).read_text()), ROOT)
    if cfg.get('schema_version') != 4:
        raise ValueError('This code-only repository supports schema-4 ablations only')
    from qrm.history.config import validate_config
    validate_config(cfg)
    benchmark = cfg['benchmark']
    if (benchmark['name'] != 'robomemarena' or benchmark['observation_protocol'] != 'policy_query_only'
            or benchmark['reward'] != 'terminal_official_tsr'):
        raise ValueError('Use the registered RoboMemArena query-time evaluation protocol')
    if cfg['memory']['coarse_dim'] != cfg['memory']['query_dim']:
        raise ValueError('The shared pooling interface requires coarse_dim == query_dim')
    partitions = [benchmark[key] for key in ('train_seeds', 'validation_seeds', 'test_seeds')]
    if any(low > high for low, high in partitions):
        raise ValueError('Invalid seed interval')
    if any(max(partitions[i][0], partitions[j][0]) <= min(partitions[i][1], partitions[j][1])
           for i in range(3) for j in range(i)):
        raise ValueError('Seed partitions overlap')
    return cfg
