"""Pin episode partitions and calculate RoboMemArena norms from train frames only."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from qrm.config import ROOT, load_config
from qrm.evaluation.manifest import sha256
from qrm.lerobot_v3 import episode_numeric


def split_episodes(records, benchmark, *, seed=11, validation_per_task=10):
    for task in benchmark["full_tasks"]:
        candidates = []
        for r in records:
            if r["task_id"] != task:
                continue
            if benchmark["test_seeds"][0] <= r["seed"] <= benchmark["test_seeds"][1]:
                r["split"] = "test"
                r["split_reason"] = "reserved_online_test_seed"
            elif benchmark["validation_seeds"][0] <= r["seed"] <= benchmark["validation_seeds"][1]:
                r["split"] = "validation"
                r["split_reason"] = "reserved_online_validation_seed"
            else:
                candidates.append(r)
        candidates.sort(key=lambda r: hashlib.sha256(f'{seed}:{task}:{r["seed"]}'.encode()).hexdigest())
        if len(candidates) <= validation_per_task:
            raise ValueError(f"Insufficient train episodes for task {task}")
        for i, r in enumerate(candidates):
            r["split"] = "validation" if i < validation_per_task else "train"
            r["split_reason"] = "deterministic_task_stratified_demo_partition"
    if any("split" not in r for r in records):
        raise ValueError("Unregistered task")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "ablation_memory_capacity/configs/k8.yaml"))
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.out is None:
        args.out = Path(cfg['dataset']['manifest']).parent
    if (args.out / 'robomemarena_lerobot_manifest.json').exists():
        raise FileExistsError('Use a new output directory; existing experiment splits must not be overwritten')
    root = Path(cfg["dataset"]["root"])
    receipt = json.loads((root / "download_manifest.json").read_text())
    if not receipt["complete"] or receipt["revision"] != cfg["dataset"]["revision"]:
        raise ValueError("A complete verified pinned download is required")
    for entry in receipt["verified_files"]:
        if sha256(root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Changed download: {entry['path']}")
    print("All source file hashes verified", flush=True)
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0":
        raise ValueError("Expected LeRobot v3.0")
    sources = {r["episode_index"]: r for r in pq.ParquetFile(root / "meta/episode_sources.parquet").read().to_pylist()}
    prompts = {r["task_index"] + 1: r["task"] for r in pq.ParquetFile(root / "meta/tasks.parquet").read().to_pylist()}
    prompt_file = Path(cfg["benchmark"]["root"]) / "evaluation_benchmark/scripts/eval_common.py"
    tree = ast.parse(prompt_file.read_text())
    official = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "TASK_PROMPTS" for t in n.targets))
    if any(prompts[t].strip() != official[f"task{t}"].strip() for t in prompts):
        raise ValueError("Dataset prompts differ from the pinned benchmark")
    records = []
    for path in sorted((root / "meta/episodes").rglob("*.parquet")):
        pf = pq.ParquetFile(path)
        cols = [c for c in pf.schema_arrow.names if not c.startswith("stats/")]
        for row in pf.read(columns=cols).to_pylist():
            source = sources[row["episode_index"]]
            task, seed = source["benchmark_task_id"], source["collection_seed"]
            record = dict(episode_id=f"task{task:02d}-seed{seed}", task_id=task, seed=seed,
                          source_episode_index=row["episode_index"], length=row["length"],
                          dataset_from_index=row["dataset_from_index"], dataset_to_index=row["dataset_to_index"],
                          data_path=f'data/chunk-{row["data/chunk_index"]:03d}/file-{row["data/file_index"]:03d}.parquet',
                          collection_bddl_stem=source["collection_bddl_stem"], prompt=prompts[task],
                          fps=info["fps"], videos={})
            for camera in ("front", "wrist"):
                key = f"videos/observation.images.{camera}"
                record["videos"][camera] = dict(
                    path=f'{key}/chunk-{row[key+"/chunk_index"]:03d}/file-{row[key+"/file_index"]:03d}.mp4',
                    from_timestamp=row[key+"/from_timestamp"], to_timestamp=row[key+"/to_timestamp"])
                if abs((row[key+"/to_timestamp"] - row[key+"/from_timestamp"]) * info["fps"] - row["length"]) > 0.1:
                    raise ValueError("Episode video duration mismatch")
            records.append(record)
    records.sort(key=lambda r: r["source_episode_index"])
    if len(records) != info["total_episodes"] or len({(r['task_id'], r['seed']) for r in records}) != len(records):
        raise ValueError("Missing or duplicate episodes")
    split_seed = cfg['dataset'].get('split_seed', 11)
    split_episodes(records, cfg["benchmark"], seed=split_seed)
    counts = {s: dict(sorted(Counter(r["task_id"] for r in records if r["split"] == s).items()))
              for s in ("train", "validation", "test")}
    document = dict(schema=2, format="lerobot_v3", root=str(root), repo_id=receipt["repo_id"],
                    revision=receipt["revision"], files=sorted(receipt["verified_files"], key=lambda e: e["path"]),
                    partition_policy=dict(seed=split_seed, validation_per_task=10, reserved_online_seeds={
                        k: cfg["benchmark"][k] for k in ("validation_seeds", "test_seeds")}),
                    prompt_source_sha256=sha256(prompt_file), counts=counts, episodes=records)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out / "robomemarena_lerobot_manifest.json"
    manifest.write_text(json.dumps(document, indent=2) + "\n")
    if cfg.get('schema_version') == 2:
        reset = dict(schema_version=2, benchmark='robomemarena', tasks=cfg['benchmark']['full_tasks'],
                     validation=list(range(200000,200010)), test=list(range(50,101)),
                     training_start=1000000, training_assignment='task round-robin; unique trial indices',
                     demonstration_exclusions=[r['episode_id'] for r in records if r['split']=='test'],
                     scene_identity_audit='pending: seed separation alone is not proof of distinct scenes')
        (args.out/'reset_manifest.json').write_text(json.dumps(reset,indent=2)+'\n')
    train = [r for r in records if r["split"] == "train"]
    states, actions = [], []
    for r in train:
        state, action = episode_numeric(dict(r, root=str(root)))
        # Copy only this episode; avoid retaining the full cached table via views.
        states.append(state.copy())
        actions.append(action.copy())
    stats = {}
    for key, pieces in (("state", states), ("actions", actions)):
        values = np.concatenate(pieces)
        q01, q99 = np.quantile(values, [0.01, 0.99], axis=0)
        stats[key] = dict(mean=values.mean(axis=0, dtype=np.float64).tolist(),
                          std=values.std(axis=0, dtype=np.float64).tolist(), q01=q01.tolist(), q99=q99.tolist())
    normdir = args.out / "robomemarena_norm"
    normdir.mkdir(exist_ok=True)
    normpath = normdir / "norm_stats.json"
    normpath.write_text(json.dumps({"norm_stats": stats}, indent=2) + "\n")
    provenance = dict(data_manifest_sha256=sha256(manifest), norm_stats_sha256=sha256(normpath),
                      split="train", episodes=len(train), frames=sum(r["length"] for r in train),
                      episode_ids=[r["episode_id"] for r in train],
                      estimator="all train frames; float64 mean/std and exact linear 1%/99% quantiles",
                      action_transform="identity; source is already delta OSC_POSE")
    (normdir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"manifest": str(manifest), "norm_stats": str(normpath),
                      "episodes_by_split": {s: sum(c.values()) for s, c in counts.items()},
                      "train_frames": provenance["frames"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
