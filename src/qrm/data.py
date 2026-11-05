"""Episode-only input contract; diagnostic labels are intentionally not represented."""
from pathlib import Path
import json
import numpy as np
from qrm.evaluation.manifest import sha256


def load_manifest(path, split="train"):
    manifest_path = Path(path).resolve()
    document = json.loads(manifest_path.read_text())
    if document.get("schema") not in {1, 2}:
        raise ValueError("Expected episode manifest schema 1 (NPZ) or 2 (LeRobot v3)")
    if document["schema"] == 2:
        if document.get("format") != "lerobot_v3":
            raise ValueError("Schema 2 requires LeRobot v3")
        root = Path(document["root"]).resolve()
        for entry in document["files"]:
            file = (root / entry["path"]).resolve()
            if not file.is_relative_to(root) or sha256(file) != entry["sha256"]:
                raise ValueError(f"Dataset hash changed: {file}")
    records = document["episodes"]
    if any(r["split"] not in {"train", "validation", "test"} for r in records):
        raise ValueError("Unknown episode partition")
    if len({r["episode_id"] for r in records}) != len(records):
        raise ValueError("Episodes cannot appear in multiple partitions")
    identities = [(r["task_id"], r["seed"]) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate task/seed identity across partitions")
    selected = []
    for record in records:
        if record["split"] != split:
            continue
        if document["schema"] == 2:
            selected.append(dict(record, root=str(root), format="lerobot_v3"))
            continue
        file = (manifest_path.parent / record["path"]).resolve()
        if sha256(file) != record["sha256"]:
            raise ValueError(f"Dataset hash changed: {file}")
        selected.append(dict(record, path=str(file)))
    if not selected:
        raise ValueError(f"No {split} episodes in manifest")
    return selected


def validate_seed_partition(records, benchmark, split):
    key = {"train":"train_seeds", "validation":"validation_seeds", "test":"test_seeds"}[split]
    low, high = benchmark[key]
    for record in records:
        if record.get("format") == "lerobot_v3":
            # Demonstration collection seeds are not the online rollout seed range.
            # Never relabel existing episodes as new simulator seeds.
            forbidden = [benchmark["test_seeds"], benchmark["validation_seeds"]]
            invalid = split == "train" and any(a <= record["seed"] <= b for a, b in forbidden)
        else:
            invalid = not low <= record["seed"] <= high
        if record["split"] != split or invalid:
            raise ValueError(f"Episode {record['episode_id']} violates the locked {split} seed range")
        if record["task_id"] not in benchmark["full_tasks"]:
            raise ValueError("Episode belongs to an unregistered task")


def decisions(record, chunk=10, stride=None):
    stride = chunk if stride is None else stride
    if chunk < 1 or stride < 1:
        raise ValueError("Action horizon and observation stride must be positive")
    if record.get("format") == "lerobot_v3":
        from qrm.lerobot_v3 import episode_decisions
        yield from episode_decisions(record, chunk, stride)
        return
    with np.load(record["path"], allow_pickle=False) as data:
        required = {"images", "wrist_images", "states", "actions", "prompt"}
        if not required.issubset(data.files):
            raise ValueError(f"Missing fields: {required-set(data.files)}")
        n = len(data["actions"])
        if data["states"].shape != (n, 8) or data["actions"].shape != (n, 7) or n == 0:
            raise ValueError("Expected one complete episode: state[T,8], action[T,7]")
        if len(data["images"]) != n or len(data["wrist_images"]) != n:
            raise ValueError("Image/action alignment error")
        prompt = str(data["prompt"].item())
        for t in range(0, n, stride):
            raw = {"observation/image": data["images"][t], "observation/wrist_image": data["wrist_images"][t],
                   "observation/state": data["states"][t], "prompt": prompt}
            # Future expert controls are supervised targets only, never part of raw/model query.
            target = data["actions"][t:min(n, t+chunk)]
            if len(target) < chunk:
                target = np.concatenate((target, np.repeat(target[-1:], chunk-len(target), axis=0)))
            yield t, raw, target
