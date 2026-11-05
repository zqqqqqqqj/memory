"""Optional W&B preflight using the receiving server's own credentials."""
import os


def check_connection():
    import wandb
    if not os.environ.get('WANDB_PROJECT'):
        raise ValueError('Set WANDB_PROJECT and authenticate with wandb login on this server')
    api = wandb.Api()
    if not api.viewer:
        raise ValueError('W&B login is required for WANDB_MODE=online')
    return dict(status='authenticated', project=os.environ['WANDB_PROJECT'])
