"""One W&B run per training stage, with durable identity and rank-zero I/O."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
import uuid


DEFAULT_ENTITY = None
DEFAULT_PROJECT = 'qrm-ablations'
_PRIVATE_KEYS = {'api_key', 'wandb_api_key', 'password', 'secret', 'credential',
                 'credentials', 'access_token', 'auth_token'}


def training_config(value):
    """Serialize explicit training metadata, excluding authentication fields."""
    if isinstance(value, dict):
        return {str(key): training_config(item) for key, item in value.items()
                if str(key).lower() not in _PRIVATE_KEYS}
    if isinstance(value, (list, tuple)):
        return [training_config(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported W&B training metadata type: {type(value).__name__}')


def numeric_metrics(value, prefix=''):
    """Only scalars are uploaded: never observations, tensors, files or videos."""
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(numeric_metrics(item, f'{prefix}/{key}' if prefix else str(key)))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(numeric_metrics(item, f'{prefix}/{index}'))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(value):
            result[prefix] = value
    return result


class TrainingTracker:
    """Fail visibly if online initialization fails; nonzero ranks never import SDK.

    Metrics use a custom global_step axis rather than W&B's transport step, so
    replay from an older retained checkpoint is accepted by the same online run.
    Local JSONL remains the canonical record of retained/discarded updates.
    """

    def __init__(self, context, out, *, stage, config, resume=False, start_step=0):
        self.context, self.out, self.stage = context, Path(out), stage
        self.config, self.resume, self.start_step = config, resume, int(start_step)
        self.run = None
        self.state = None
        self.mode = os.environ.get('WANDB_MODE', 'disabled')
        if self.mode not in {'online', 'offline', 'disabled'}:
            raise ValueError('WANDB_MODE must be online, offline or disabled')

    @property
    def state_path(self):
        return self.out/'wandb_run.json'

    def _write_state(self):
        temporary = self.state_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.state, indent=2)+'\n')
        temporary.replace(self.state_path)

    def __enter__(self):
        error = None
        if self.context.primary:
            try:
                self._start()
            except BaseException as exc:
                error = type(exc).__name__
                if self.run is not None:
                    try:
                        self.run.finish(exit_code=1)
                    except Exception:
                        pass
                if self.state is not None:
                    self.state.update(status='initialization_failed', error_type=error)
                    self._write_state()
        error = self.context.broadcast(error)
        if error is not None:
            raise RuntimeError(f'W&B {self.mode} initialization failed ({error}); '
                               'training stopped; check credentials/connectivity and local W&B logs') from None
        return self

    def _start(self):
        self.out.mkdir(parents=True, exist_ok=True)
        entity = os.environ.get('WANDB_ENTITY', DEFAULT_ENTITY)
        project = os.environ.get('WANDB_PROJECT', DEFAULT_PROJECT)
        if self.state_path.exists():
            if not self.resume:
                raise FileExistsError('W&B identity exists; use the training --resume option')
            self.state = json.loads(self.state_path.read_text())
            for key, expected in dict(entity=entity, project=project, stage=self.stage, mode=self.mode).items():
                if self.state.get(key) != expected:
                    raise ValueError(f'W&B resume identity changed: {key}')
            self.state['resume_count'] = self.state.get('resume_count', 0)+1
        else:
            if self.resume and self.mode != 'disabled':
                raise FileNotFoundError('Cannot resume W&B without this output directory wandb_run.json')
            self.state = dict(schema=1, run_id='qrm-history-'+uuid.uuid4().hex[:16],
                              entity=entity, project=project, stage=self.stage, mode=self.mode,
                              resume_count=0, created_at=time.time())
        previously_connected = bool(self.state.get('connected_at'))
        self.state.update(status='initializing', resume_from_step=self.start_step,
                          last_step=self.start_step, updated_at=time.time())
        self._write_state()
        if self.mode == 'disabled':
            self.state.update(status='disabled')
            self._write_state()
            return
        import wandb
        directory = self.out/'wandb'
        directory.mkdir(exist_ok=True)
        # Explicit mode prevents W&B from silently falling back to disabled.
        self.run = wandb.init(entity=entity, project=project, id=self.state['run_id'],
            name=self.out.name, job_type=self.stage,
            group=os.environ.get('WANDB_RUN_GROUP', 'history-sft'),
            mode=self.mode, resume='must' if self.resume and previously_connected else 'allow',
            dir=str(directory), config=training_config(self.config),
            settings=wandb.Settings(console='off', disable_git=True, save_code=False,
                x_disable_stats=True, x_disable_meta=True, x_save_requirements=False, init_timeout=120))
        if self.run is None or (self.mode == 'online' and
                               (getattr(self.run, 'offline', False) or getattr(self.run, 'disabled', False))):
            raise RuntimeError('W&B did not establish the required online run')
        self.run.define_metric('global_step')
        for namespace in ('train', 'validation', 'checkpoint', 'events'):
            self.run.define_metric(namespace+'/*', step_metric='global_step')
        self.state.update(status='connected', url=self.run.url, connected_at=time.time())
        self._write_state()
        self.run.summary['training_stage'] = self.stage
        self.run.summary['training_status'] = 'running'
        self.run.summary['resume_count'] = self.state['resume_count']
        self.run.summary['resume_from_step'] = self.start_step
        if self.resume:
            self.run.log(dict(global_step=self.start_step,
                **{'events/resume_from_step': self.start_step,
                   'events/resume_count': self.state['resume_count']}))
        print(f'W&B {self.mode}: {self.run.url}', flush=True)

    def log(self, row):
        if not self.context.primary:
            return
        step = int(row['step'])
        if self.run is not None:
            validation = row.get('validation', {})
            train = {key: value for key, value in row.items() if key not in {'step', 'validation'}}
            payload = dict(global_step=step, **numeric_metrics(train, 'train'),
                           **numeric_metrics(validation, 'validation'))
            self.run.log(payload)
            self.run.summary['last_training_step'] = step
        self.state.update(last_step=step, updated_at=time.time())
        self._write_state()

    def checkpoint(self, step, *, save_seconds):
        if self.context.primary and self.run is not None:
            self.run.log(dict(global_step=int(step), **{'checkpoint/saved_step': int(step),
                                                       'checkpoint/save_seconds': float(save_seconds)}))
            self.run.summary['last_checkpoint_step'] = int(step)

    def __exit__(self, error_type, error, traceback):
        if self.context.primary and self.state is not None:
            status = 'completed' if error is None else ('interrupted' if isinstance(error, KeyboardInterrupt)
                                                       else getattr(error, 'training_status', 'failed'))
            self.state.update(status=status, updated_at=time.time())
            if error_type:
                self.state['error_type'] = error_type.__name__
            self._write_state()
            if self.run is not None:
                self.run.summary['training_status'] = status
                self.run.summary['last_training_step'] = self.state['last_step']
                try:
                    self.run.finish(exit_code=0 if error is None else 1)
                except Exception:
                    self.state.update(status='sync_failed')
                    self._write_state()
                    if error is None:
                        raise RuntimeError('W&B final synchronization failed; local training logs were retained') from None
                    print('W&B final synchronization failed; original training failure and local logs retained', flush=True)
        return False
