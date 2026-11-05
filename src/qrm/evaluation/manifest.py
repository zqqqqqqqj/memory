from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import os


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(config, out, *, config_path, role, extra=None):
    root = Path(config["benchmark"]["root"])
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != config["benchmark"]["commit"]:
        raise RuntimeError("Benchmark commit changed; explicitly review and update the protocol")
    openpi = Path(config["model"]["openpi_root"])
    openpi_commit = subprocess.check_output(["git", "-C", str(openpi), "rev-parse", "HEAD"], text=True).strip()
    if openpi_commit != config["model"]["openpi_commit"]:
        raise RuntimeError("OpenPI commit changed; review the backbone integration")
    files = list((root/"evaluation_benchmark/scripts").glob("*.py"))
    files += list((root/"evaluation_benchmark/bddl").glob("*.bddl"))
    packages = {}
    for name in ("torch", "numpy", "mujoco", "robosuite", "transformers", "jax", "av", "pyarrow", "lerobot"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    report = dict(role=role, python=platform.python_version(), config=config,
                  config_sha256=sha256(config_path), benchmark_commit=commit,
                  scoring_and_task_hashes={str(p.relative_to(root)): sha256(p) for p in files},
                  openpi_commit=openpi_commit,
                  project_source_hashes={str(p.relative_to(Path(__file__).resolve().parents[3])):sha256(p)
                    for base in ("src", "scripts")
                    for p in (Path(__file__).resolve().parents[3]/base).rglob("*")
                    if p.is_file() and p.suffix in {'.py', '.sh'}},
                  dependency_lock_hashes={str(p):sha256(p) for p in
                    [openpi/'uv.lock', * (Path(__file__).resolve().parents[3]/'configs').glob('requirements*.txt')]
                    if p.is_file()},
                  packages=packages, gpu_selection={k:os.environ.get(k) for k in
                    ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")}, extra=extra or {})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    return report
