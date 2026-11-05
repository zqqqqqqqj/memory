"""Read the pinned RoboMemArena LeRobot v3 tables/videos without expanding images.

Only RGB, proprioception, the task instruction and supervised actions are exposed.
Episode video offsets are independent of global dataset row offsets.
"""
from contextlib import closing
from functools import lru_cache
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq


@lru_cache(maxsize=1)
def numeric_file(path):
    table = pq.ParquetFile(path).read(columns=[
        "observation.state", "action", "episode_index", "frame_index", "index", "task_index"])
    result = {}
    for name in table.column_names:
        a = table[name].combine_chunks()
        if name in {"observation.state", "action"}:
            width = 8 if name == "observation.state" else 7
            if not np.all(a.value_lengths().to_numpy() == width):
                raise ValueError(f"Wrong dimension: {name}")
            result[name] = a.flatten().to_numpy().reshape(-1, width)
        else:
            result[name] = a.to_numpy()
    return result


def episode_numeric(record):
    values = numeric_file(str(Path(record["root"]) / record["data_path"]))
    start = int(np.searchsorted(values["index"], record["dataset_from_index"]))
    end = start + record["length"]
    out = {k: v[start:end] for k, v in values.items()}
    if (len(out["index"]) != record["length"]
            or not np.array_equal(out["index"], np.arange(record["dataset_from_index"], record["dataset_to_index"]))
            or not np.all(out["episode_index"] == record["source_episode_index"])
            or not np.array_equal(out["frame_index"], np.arange(record["length"]))
            or not np.all(out["task_index"] + 1 == record["task_id"])):
        raise ValueError(f"Episode boundary/identity mismatch: {record['episode_id']}")
    if not all(np.isfinite(out[k]).all() for k in ("observation.state", "action")):
        raise ValueError("Nonfinite demonstration values")
    return out["observation.state"], out["action"]


def video_frames(path, frame_indices, fps):
    """Seek once to the preceding keyframe, decode monotonically, match exact PTS."""
    wanted = iter(frame_indices)
    target = next(wanted, None)
    if target is None:
        return
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        if abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError(f"Unexpected video FPS: {path}")
        container.seek(int(target / fps / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            position = float(frame.pts * stream.time_base) * fps
            index = round(position)
            if abs(position - index) > 0.1:
                raise ValueError("Video PTS does not align with control frame grid")
            if index < target:
                continue
            if index > target:
                raise ValueError(f"Missing video frame {target} in {path}; got {index}")
            yield frame.to_ndarray(format="rgb24")
            previous = target
            target = next(wanted, None)
            if target is None:
                return
            if target <= previous:
                raise ValueError("Video frame requests must increase strictly")
        raise ValueError(f"Video ended before requested frame {target}: {path}")


def episode_decisions(record, chunk, stride, *, timesteps=None):
    states, actions = episode_numeric(record)
    n = len(actions)
    times = list(range(0, n, stride) if timesteps is None else timesteps)
    if any(t < 0 or t >= n for t in times):
        raise ValueError("Requested timestep outside episode")
    streams = []
    for camera in ("front", "wrist"):
        video = record["videos"][camera]
        offset = video["from_timestamp"] * record["fps"]
        if abs(offset - round(offset)) > 0.1:
            raise ValueError("Episode video offset is not frame aligned")
        streams.append(video_frames(Path(record["root"]) / video["path"],
                                    [round(offset) + t for t in times], record["fps"]))
    with closing(streams[0]), closing(streams[1]):
        for t, front, wrist in zip(times, *streams, strict=True):
            raw = {"observation/image": front, "observation/wrist_image": wrist,
                   "observation/state": states[t], "prompt": record["prompt"]}
            # Delta OSC controls are already deltas. Do not difference them again.
            target = actions[t:min(n, t + chunk)]
            if len(target) < chunk:
                target = np.concatenate([target, np.repeat(target[-1:], chunk - len(target), axis=0)])
            yield t, raw, target
