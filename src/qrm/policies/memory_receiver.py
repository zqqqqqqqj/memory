from contextlib import contextmanager

@contextmanager
def injected_prefix(model, prepared_prefix, preprocessing, receiver, memory):
    """Single-call injection; restores methods and retains no cross-step KV/prefix."""
    previous_prefix, previous_preprocess = model.embed_prefix, model._preprocess_observation
    try:
        model.embed_prefix = lambda *args, **kwargs: receiver(*prepared_prefix, memory)
        model._preprocess_observation = lambda *args, **kwargs: preprocessing
        yield
    finally:
        model.embed_prefix, model._preprocess_observation = previous_prefix, previous_preprocess
