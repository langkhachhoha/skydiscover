"""A/B wall-clock test: gpt-5 non-reasoning ("minimal") vs reasoning effort, on the OpenEvolve backend.

gpt-5 has no true "non-reasoning" switch; the closest thing is reasoning_effort="minimal",
which asks the model for (near-)zero reasoning tokens before answering. This script runs the
same OpenEvolve search once per effort level and reports wall-clock time, per-iteration LLM
latency, reasoning-token usage, and best score, so speed can be traded off against quality.

Usage:
    python scripts/ablation/reasoning_effort_ab.py \
        --benchmark benchmarks/math/circle_packing \
        --iterations 10 --efforts minimal low medium

Requires OPENAI_API_KEY (OpenRouter key works) in the environment or in .env.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def make_config(base_config: Path, effort: str, out_dir: Path, iterations: int) -> Path:
    """Copy the benchmark config, forcing llm.reasoning_effort and disabling the monitor."""
    cfg = yaml.safe_load(base_config.read_text())
    cfg["max_iterations"] = iterations
    llm = cfg.setdefault("llm", {})
    if effort == "default":
        llm.pop("reasoning_effort", None)
    else:
        llm["reasoning_effort"] = effort
    for model in llm.get("models", []):
        model.pop("reasoning_effort", None)
    cfg.setdefault("monitor", {})["enabled"] = False
    cfg["human_feedback_enabled"] = False
    path = out_dir / f"config_{effort}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def parse_best_score(run_dir: Path) -> float | None:
    """Best combined score from OpenEvolve's best_program_info.json, if written."""
    for candidate in sorted(run_dir.rglob("best_program_info.json")):
        try:
            info = json.loads(candidate.read_text())
        except Exception:
            continue
        metrics = info.get("metrics", {})
        for key in ("combined_score", "score", "sum_radii"):
            if key in metrics:
                return float(metrics[key])
    return None


def run_one(benchmark: Path, effort: str, iterations: int, out_root: Path) -> dict:
    run_dir = out_root / effort
    run_dir.mkdir(parents=True, exist_ok=True)
    config = make_config(benchmark / "config.yaml", effort, run_dir, iterations)
    cmd = [
        sys.executable,
        "-m",
        "skydiscover.cli",
        str(benchmark / "initial_program.py"),
        str(benchmark / "evaluator.py"),
        "-c",
        str(config),
        "-o",
        str(run_dir / "out"),
        "-i",
        str(iterations),
        "-s",
        "openevolve",
        "-l",
        "INFO",
    ]
    log_path = run_dir / "run.log"
    start = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start

    log_text = log_path.read_text()
    reasoning_tokens = sum(int(m) for m in re.findall(r"reasoning_tokens[\"']?\s*[:=]\s*(\d+)", log_text))
    # OpenEvolve swallows per-iteration LLM failures and still exits 0, which would make a
    # run that never reached the API look like a very fast one. Flag those explicitly.
    llm_failures = log_text.count("LLM generation failed")
    return {
        "effort": effort,
        "iterations": iterations,
        "wall_clock_s": round(elapsed, 1),
        "s_per_iteration": round(elapsed / max(iterations, 1), 1),
        "reasoning_tokens_logged": reasoning_tokens or None,
        "best_score": parse_best_score(run_dir),
        "exit_code": proc.returncode,
        "llm_failures": llm_failures,
        "valid": proc.returncode == 0 and llm_failures == 0,
        "log": str(log_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmarks/math/circle_packing")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--efforts",
        nargs="+",
        default=["minimal", "medium"],
        help="reasoning_effort values to compare; 'default' leaves the field unset",
    )
    parser.add_argument("--output", default="outputs/reasoning_effort_ab")
    args = parser.parse_args()

    load_dotenv(REPO / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    benchmark = (REPO / args.benchmark).resolve()
    out_root = (REPO / args.output).resolve() / time.strftime("%Y%m%d-%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for effort in args.efforts:
        print(f"=== running effort={effort} ({args.iterations} iterations) ===", flush=True)
        row = run_one(benchmark, effort, args.iterations, out_root)
        rows.append(row)
        print(json.dumps(row), flush=True)

    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\neffort      wall_clock   s/iter   best_score   valid")
    for row in rows:
        print(
            f"{row['effort']:<10} {row['wall_clock_s']:>9.1f}s {row['s_per_iteration']:>8.1f} "
            f"{str(row['best_score']):>12} {str(row['valid']):>7}"
        )
    if not all(row["valid"] for row in rows):
        print("\nWARNING: some runs had failed LLM calls — their timings are meaningless.")
    print(f"\nsummary: {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
