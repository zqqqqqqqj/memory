#!/usr/bin/env bash
# First activate the pinned OpenPI Python environment on this server.
export QRM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${OPENPI_ROOT:-}" ] && [ -f "${QRM_SERVER_CONFIG:-$QRM_ROOT/server.local.yaml}" ]; then
  OPENPI_ROOT="$(python -c 'import os,yaml; from pathlib import Path; p=Path(os.environ.get("QRM_SERVER_CONFIG",str(Path(os.environ["QRM_ROOT"])/"server.local.yaml"))); v=Path(os.path.expandvars(yaml.safe_load(p.read_text())["model"]["openpi_root"])).expanduser(); print(v.resolve() if v.is_absolute() else (p.resolve().parent/v).resolve())')"
  export OPENPI_ROOT
fi
export PYTHONPATH="$QRM_ROOT/src:${OPENPI_ROOT:-$QRM_ROOT/vendor/openpi}/src:${OPENPI_ROOT:-$QRM_ROOT/vendor/openpi}/packages/openpi-client/src:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_PROJECT="${WANDB_PROJECT:-qrm-ablations}"
export WANDB_DIR="$QRM_ROOT/runs"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
