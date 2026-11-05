"""Server-local path overrides; experiment hyperparameters cannot be overridden here."""
import os
from pathlib import Path

import yaml


PATH_FIELDS = {
    'dataset': {'root', 'manifest', 'norm_stats'},
    'benchmark': {'root'},
    'model': {'openpi_root', 'base_checkpoint_pytorch', 'base_checkpoint_jax'},
}


def apply_server_paths(cfg, root):
    specified = os.environ.get('QRM_SERVER_CONFIG')
    path = Path(specified).expanduser() if specified else Path(root)/'server.local.yaml'
    if not path.is_file():
        if specified:
            raise FileNotFoundError(f'QRM_SERVER_CONFIG does not exist: {path}')
        return cfg
    overrides = yaml.safe_load(path.read_text())
    if not isinstance(overrides, dict) or not set(overrides) <= PATH_FIELDS.keys():
        raise ValueError('server.local.yaml may contain only dataset, benchmark and model path sections')
    for section, entries in overrides.items():
        if not isinstance(entries, dict) or not set(entries) <= PATH_FIELDS[section]:
            raise ValueError(f'Unsupported local override in {section}; training settings must stay in experiment YAML')
        for key, value in entries.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f'Expected a local path for {section}.{key}')
            resolved = Path(os.path.expandvars(value)).expanduser()
            if not resolved.is_absolute():
                resolved = path.resolve().parent/resolved
            cfg[section][key] = str(resolved.resolve())
    return cfg
