"""Publish validation decisions before any training rank may exit."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

from .validation import atomic_json


def resume_training_validation(out, checkpoint, cfg, context, step):
    """Finish the restored checkpoint's validation before advancing training.

    A checkpoint is published before its rollout validation. A restart may
    therefore find missing or failed validation even though the weights are
    complete. Reuse only independently re-collected results plus their exact
    gate. Preserve incomplete attempts before rerunning; mismatched provenance
    or corrupted completed results fail rather than being silently replaced.
    All preparation decisions/errors are broadcast before any rank proceeds.
    """
    from . import validation as v

    if step <= 0 or (step % cfg['sft']['eval_every'] and step != cfg['sft']['steps']):
        return None
    directory = Path(out)/'validation'/f'step_{step:06d}'
    prepared = None
    if context.primary:
        try:
            metadata = v.checkpoint_metadata(checkpoint, cfg)
            if metadata['checkpoint_step'] != step:
                raise ValueError('Restored checkpoint step differs from resume validation')
            expected = v.build_validation_plan(cfg, metadata, context.world_size)
            if directory.is_symlink():
                raise ValueError('Resume validation output must not be a symlink')
            saved = {}
            for name in ('validation_plan.json', 'config.json', 'status.json', 'metrics.json', 'gate.json'):
                path = directory/name
                if path.exists():
                    saved[name] = json.loads(path.read_text())
            if 'validation_plan.json' in saved and saved['validation_plan.json'] != expected:
                raise ValueError('Saved resume validation plan/checkpoint/source identity changed')
            if 'config.json' in saved and saved['config.json'] != cfg:
                raise ValueError('Saved resume validation config changed')
            if 'status.json' in saved and saved['status.json'].get('checkpoint_step') != step:
                raise ValueError('Saved resume validation status belongs to another step')
            if 'metrics.json' in saved and any(saved['metrics.json'].get(key) != expected[key]
                    for key in metadata):
                raise ValueError('Saved resume validation metrics provenance changed')
            complete = (len(saved) == 5 and saved['status.json'] ==
                        dict(status='completed', checkpoint_step=step))
            if complete:
                try:
                    report = v.collect_validation_report(directory, expected)
                except FileNotFoundError:
                    complete = False
                else:
                    gate = v.validation_gate(report, cfg, step)
                    if saved['metrics.json'] != report or saved['gate.json'] != gate:
                        raise ValueError('Completed resume validation metrics/gate do not match episode evidence')
                    prepared = dict(mode='reused', report=report, gate=gate, error=None)
            if not complete:
                previous = None
                if directory.exists():
                    previous = directory.parent/'previous_attempts'/f'{directory.name}_{time.time_ns()}'
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    directory.rename(previous)
                    for parent in (directory.parent, previous.parent):
                        descriptor = os.open(parent, os.O_RDONLY)
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                prepared = dict(mode='reevaluated', previous_attempt=str(previous) if previous else None,
                                error=None)
        except Exception as error:
            prepared = dict(error=f'{type(error).__name__}: {error}')
    prepared = context.broadcast(prepared)
    if prepared['error']:
        raise RuntimeError('Resume validation preparation failed: '+prepared['error'])
    if prepared['mode'] == 'reevaluated':
        report = v.evaluate_history_checkpoint(checkpoint, cfg, context, directory)
        prepared.update(report=report, gate=v.validation_gate(report, cfg, step))
    return dict(directory=directory, **prepared)


def publish_training_validation(out, validation_out, report, gate, context, tracker, step):
    """All ranks wait for primary publication, including ordinary logging errors.

    A quality rejection is already durable before external tracking is called.
    The trainer separately coordinates its final exit so torchrun cannot kill
    rank zero while it finishes the W&B run.
    """
    error = None
    if context.primary:
        try:
            out, validation_out = Path(out), Path(validation_out)
            atomic_json(validation_out/'gate.json', gate)
            record = dict(step=step, validation=report, gate=gate)
            with (out/'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            if not gate['passed']:
                atomic_json(out/'training.json', dict(status='validation_rejected', step=step,
                    pid=os.getpid(), updated_unix=time.time(), error_type='ValidationRejected',
                    reason=f"Validation macro TSR {report['macro_TSR']:.4f} did not exceed {gate['min_macro_tsr']}",
                    validation=str(validation_out/'metrics.json'), gate=gate))
            tracker.log(dict(step=step, validation={**report, 'gate': gate}))
            print(json.dumps(record, allow_nan=False), flush=True)
        except Exception as exception:
            error = f'{type(exception).__name__}: {exception}'
            try:
                atomic_json(Path(out)/'training.json', dict(status='failed', step=step,
                    pid=os.getpid(), updated_unix=time.time(), error_type='ValidationPublicationError', reason=error))
            except OSError:
                pass
    error = context.broadcast(error)
    if error is not None:
        raise RuntimeError('Validation publication failed: '+error)
