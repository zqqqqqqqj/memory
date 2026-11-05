"""Exact ordered inputs: consult frozen-feature cache before decoding history.

Only deterministic CPU work runs ahead. Frozen vision, all trainable modules,
flow RNG and CUDA operations stay in the consumer/training thread.
"""
from contextlib import closing
from dataclasses import dataclass
import hashlib
import time

import numpy as np
import torch

from qrm.data import decisions
from qrm.lerobot_v3 import episode_decisions
from qrm.prefetch import PrefetchedInputs
from .model import PrefixSample
from .visual_batch import encode_visual_chunk
from .sampling import select_decision


@dataclass
class CPUHistory:
    identity: tuple
    times: list
    visuals: list
    inputs: dict
    proprio: np.ndarray
    metrics: dict


def cached_visual(cache, identity):
    if cache is None:
        return None
    if hasattr(cache, 'lookup'):
        return cache.lookup(identity)
    key = hashlib.sha256(repr(identity).encode()).hexdigest()
    try:
        # Identical key, format and version to VisualCache.get; no new namespace.
        return torch.load(cache.root/(key+'.pt'), map_location='cpu', weights_only=True)
    except FileNotFoundError:
        return None


def prepare_cpu_history(bridge, record, selected_time, stride, cache=None, current_only=False):
    start = time.monotonic()
    if stride < 1 or not 0 <= selected_time < record['length'] or selected_time % stride:
        raise ValueError('Selected query outside decision grid')
    times = [selected_time] if current_only else list(range(0, selected_time+1, stride))
    visuals = [cached_visual(cache, (record['episode_id'], t)) for t in times]
    missing = {t for t, visual in zip(times, visuals, strict=True) if visual is None}
    # Current raw state, instruction and expert targets are always read fresh.
    needed = sorted(missing | {selected_time})
    inputs, proprio = {}, None
    if record.get('format') == 'lerobot_v3':
        stream = episode_decisions(record, bridge.horizon, stride, timesteps=needed)
    else:
        stream = decisions(record, bridge.horizon, stride)
    with closing(stream):
        for t, raw, actions in stream:
            if t not in missing and t != selected_time:
                continue
            inputs[t] = bridge.transform_cpu(raw, actions if t == selected_time else None)
            if t == selected_time:
                proprio = np.array(raw['observation/state'], dtype=np.float32, copy=True)
                break
    if proprio is None or set(inputs) != set(needed):
        raise ValueError('Missing prefix observations')
    return CPUHistory((record['episode_id'], selected_time), times, visuals, inputs, proprio,
                      dict(cpu_prepare_seconds=time.monotonic()-start,
                           cache_hits=len(times)-len(missing), cache_misses=len(missing),
                           prepared_image_decisions=len(needed)))


def materialize_history(bridge, history, cache=None, visual_encoder=None, vision_batch_size=1,
                        vision_guard=None):
    """Keep full ordered history; batch at most two missing frozen observations."""
    if vision_batch_size not in (1, 2):
        raise ValueError('Only frozen vision batch sizes 1 and 2 are validated')
    visuals = list(history.visuals)
    observation = prep = normalized = None
    calls_before = visual_encoder.vision_encoder_calls if visual_encoder is not None else 0
    reuses_before = visual_encoder.padding_visual_reuses if visual_encoder is not None else 0
    pending = []

    def flush():
        preps = [p for _, _, p in pending]
        computed = (vision_guard.encode(preps, visual_encoder,
                                        identities=[(history.identity[0], t) for _, t, _ in pending])
                    if vision_guard is not None else encode_visual_chunk(bridge, preps, visual_encoder))
        for (i, t, _), values in zip(pending, computed, strict=True):
            visual = (cache.get((history.identity[0], t), lambda: values, 'cpu')
                      if cache is not None else values)
            visuals[i] = tuple(v.cpu() for v in visual)
        pending.clear()

    for i, (t, visual) in enumerate(zip(history.times, history.visuals, strict=True)):
        # Another rank may have filled a missing entry since CPU lookahead.
        if visual is None and cache is not None:
            visual = cached_visual(cache, (history.identity[0], t))
        if visual is None or t == history.identity[1]:
            current_observation, current_actions = bridge.transform_prepared(history.inputs[t])
            current_prep = bridge.model._preprocess_observation(current_observation, train=False)
            if visual is None:
                pending.append((i, t, current_prep))
                active_batch = vision_guard.batch_size if vision_guard is not None else vision_batch_size
                if len(pending) == active_batch:
                    flush()
            if t == history.identity[1]:
                observation, normalized, prep = current_observation, current_actions, current_prep
        if visual is not None:
            visuals[i] = tuple(v.cpu() for v in visual)
    if pending:
        flush()
    proprio = torch.as_tensor(history.proprio, device=bridge.device)[None]
    if visual_encoder is not None:
        history.metrics.update(vision_encoder_calls=visual_encoder.vision_encoder_calls-calls_before,
                               padding_visual_reuses=visual_encoder.padding_visual_reuses-reuses_before)
    return PrefixSample(visuals, history.times, observation, prep, proprio, normalized, history.identity)


class PrefixRequests:
    """Stateless global indices make discarded lookahead safe across resume."""
    def __init__(self, records, seed, first_index, stop_index, world, stride):
        self.records, self.seed = records, seed
        self.next_index, self.stop_index = first_index, stop_index
        self.world, self.stride = world, stride

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_index >= self.stop_index:
            raise StopIteration
        index = self.next_index
        record, t = select_decision(self.records, self.seed, index, self.stride)
        self.next_index += self.world
        return index, t, record, t

    def state_dict(self):
        return dict(next_global_index=self.next_index)

    def close(self):
        pass


def prefetch_histories(bridge, records, seed, first_index, stop_index, world, stride,
                       cache, current_only=False, capacity=1):
    requests = PrefixRequests(records, seed, first_index, stop_index, world, stride)
    return PrefetchedInputs(requests,
                            lambda record, t: prepare_cpu_history(bridge, record, t, stride,
                                                                 cache, current_only),
                            capacity=capacity)
