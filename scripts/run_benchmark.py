import argparse
from pathlib import Path
import json
from qrm.config import ROOT, load_config
from qrm.benchmarks.robomemarena import run_episode, ZeroPolicy, RemotePolicy
from qrm.evaluation.manifest import write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT/"ablation_memory_capacity/configs/k8.yaml"))
    parser.add_argument("--tasks", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--zero-policy", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/"episodes.json").exists():
        raise FileExistsError("Use a new output directory to preserve prior results")
    tasks = args.tasks or cfg["benchmark"]["p0_tasks"]
    write_manifest(cfg, args.out/"versions.json", config_path=args.config, role="benchmark",
                   extra={"tasks": tasks, "seed": args.seed, "max_steps_override": args.max_steps,
                          "zero_policy": args.zero_policy})
    policy = ZeroPolicy() if args.zero_policy else RemotePolicy(args.host, args.port)
    results = []
    try:
        for task in tasks:
            result = run_episode(cfg, task, args.seed, policy, args.out/"videos", max_steps=args.max_steps)
            results.append(result)
            (args.out/"episodes.json").write_text(json.dumps(results, indent=2)+"\n")
            print(f"task={task} status={result['status']} steps={result['native_env_steps']} TSR={result['TSR']}", flush=True)
            if result["status"] != "completed":
                raise RuntimeError(result["errors"])
    finally:
        if not args.zero_policy:
            policy.close()


if __name__ == "__main__":
    main()
