"""Schema-4 action-policy checkpoint loader."""
from qrm.history.runtime import load_history_policy


def load_joint_policy(checkpoint, *, device='cuda:0', payload=None):
    return load_history_policy(checkpoint, device=device, payload=payload)
