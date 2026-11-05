"""Checkpoint schedules shared by history pretraining and action SFT."""


def validate_retention(training):
    policy = training.get('checkpoint_retention', 'all')
    if policy not in {'all', 'milestones_and_latest', 'latest_then_milestones'}:
        raise ValueError('Unknown checkpoint retention policy')
    save = training.get('save_every')
    if save is not None and (type(save) is not int or save < 1):
        raise ValueError('Checkpoint save interval must be a positive integer')
    if policy == 'all':
        return
    keep = training.get('keep_every')
    if type(keep) is not int or keep < 1 or (save is not None and keep % save):
        raise ValueError('Permanent checkpoint interval must be a positive multiple of save interval')
    if policy == 'latest_then_milestones':
        start = training.get('keep_from')
        if type(start) is not int or start < 0 or start % keep:
            raise ValueError('Permanent checkpoint start must align with the keep interval')


def keep_checkpoint(step, latest_step, training):
    """Keep the newest snapshot, future steps, and configured permanent steps."""
    validate_retention(training)
    policy = training.get('checkpoint_retention', 'all')
    if step >= latest_step or policy == 'all':
        return True
    start = training['keep_from'] if policy == 'latest_then_milestones' else 0
    return step >= start and step % training['keep_every'] == 0
