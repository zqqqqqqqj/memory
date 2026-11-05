"""Bind training-only normalization to a specific demonstration manifest."""
import json
from pathlib import Path
from qrm.evaluation.manifest import sha256


def validate_norm_stats(path, manifest):
    path = Path(path).resolve()
    provenance = json.loads((path.parent / "provenance.json").read_text())
    if (provenance.get("split") != "train"
            or provenance.get("data_manifest_sha256") != sha256(manifest)
            or provenance.get("norm_stats_sha256") != sha256(path)):
        raise ValueError("Normalization must come from this manifest's training episodes only")
    document = json.loads(Path(manifest).read_text())
    expected = [r for r in document["episodes"] if r["split"] == "train"]
    if (set(provenance["episode_ids"]) != {r["episode_id"] for r in expected}
            or provenance["episodes"] != len(expected)):
        raise ValueError("Normalization episode partition mismatch")
    return {"norm_stats": str(path), "norm_stats_sha256": sha256(path)}


def checkpoint_norm_stats(checkpoint):
    path = checkpoint.get("norm_stats")
    if path is not None and sha256(path) != checkpoint.get("norm_stats_sha256"):
        raise ValueError("Normalization changed since the checkpoint was trained")
    return path
