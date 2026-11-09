#!/usr/bin/env python3
"""Helpers for run_coral_cp.sh / run_coral_pk.sh: build a CORAL job directory, and report on runs.

    coral_tools.py prepare --task circle_packing --job-dir D --coral-dir C --port 4101 \
                           --model openrouter/openai/gpt-5 --agents 1
                    (tasks: circle_packing, circle_packing_rect, polyominoes, kernel_builder)
    <lane task dir lines> | coral_tools.py status [--max-evals 100]
    <lane task dir lines> | coral_tools.py compare --out DIR
    coral_tools.py count --job-dir D      # graded attempts of D's latest run (the lane's eval cap)

``prepare`` copies CORAL's own task (examples/math/<task>: seed, grader,
task text) into D and rewrites its task.yaml for this server: OpenCode agents
whose model traffic goes through CORAL's LiteLLM gateway to OpenRouter, a
local (non-tmux, non-docker) session, results under D/results.

``status`` / ``compare`` read, per job, CORAL's attempts
(.coral/public/attempts/*.json, graded by CORAL's grader) and the gateway's
request log (.coral/public/gateway/requests.jsonl: one line per LLM call,
with the provider's token usage), and turn usage into dollars with
OpenRouter's price list for the model (or OpenRouter's own usage.cost when the
response carries it). The log is read incrementally (cached offset), so
status stays fast as the log grows.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Best known sums of radii; CORAL's graders score sum_radii / BEST.
BEST_KNOWN = {"circle_packing": 2.635977, "circle_packing_rect": 2.3658321334167627}
KERNEL_BASELINE_CYCLES = 147_734

# Our task name -> where CORAL keeps it, which way its grader's score points,
# and how that score becomes the number the other methods report ("best"):
#   circle packing  CORAL score = sum_radii / best known  -> best = sum of radii
#   polyominoes     CORAL score = 0..100 (local judge)    -> best = the same score
#   kernel_builder  CORAL score = cycles (minimize)       -> best = 147734 / cycles, as in run_pk_queue.sh
TASKS = {
    "circle_packing": {"src": "examples/math/circle_packing", "direction": "maximize",
                       "best": lambda s: s * BEST_KNOWN["circle_packing"], "raw": "ratio"},
    "circle_packing_rect": {"src": "examples/math/circle_packing_rect", "direction": "maximize",
                            "best": lambda s: s * BEST_KNOWN["circle_packing_rect"], "raw": "ratio"},
    "polyominoes": {"src": "examples/frontier_cs_algo/0", "direction": "maximize",
                    "best": lambda s: s, "raw": "score"},
    "kernel_builder": {"src": "examples/kernel_builder", "direction": "minimize",
                       "best": lambda c: KERNEL_BASELINE_CYCLES / c if c > 0 else None, "raw": "cycles"},
}
DEFAULT_FRONTIER_CS_DIR = "/home/ubuntu/projects/Frontier-CS"
POLYOMINO_JUDGE = Path(__file__).resolve().parents[2] / "benchmarks" / "coral_tasks" / "polyominoes" / "judge.py"

# USD per token. Overridden by $GROUP_DIR/pricing.json (fetched from OpenRouter at start).
DEFAULT_PRICES = {
    "openrouter/openai/gpt-5": {"prompt": 1.25e-6, "completion": 1.0e-5, "input_cache_read": 1.25e-7},
}


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

# Upstream grader bugs, fixed in the job's copy only (CORAL's checkout is left
# alone). Each entry: task -> [(file under grader/, broken text, fixed text, why)].
GRADER_PATCHES = {
    "circle_packing_rect": [(
        "src/circle_packing_rect_grader/grader.py",
        'expected ({{}}, 3)".format(NUM_CIRCLES)',
        'expected ({{NUM_CIRCLES}}, 3)"',
        "CORAL 0123dfb's eval script has f\"...({}, 3)\".format(...), an empty f-string "
        "expression: a SyntaxError on Python 3.12, so every evaluation failed",
    )],
}


def _patch_grader(task: str, grader_dir: Path) -> list[str]:
    applied = []
    for rel, broken, fixed, why in GRADER_PATCHES.get(task, []):
        f = grader_dir / rel
        text = f.read_text()
        if broken in text:
            f.write_text(text.replace(broken, fixed))
            applied.append(f"{rel}: {why}")
    return applied


# CORAL's Polyominoes grader hands the solution to the frontier_cs package,
# whose judge is a go-judge server in Docker (none on this server). This one
# scores it with benchmarks/coral_tasks/polyominoes/judge.py instead -- the
# judge the baselines and SpecEvo are scored by (Frontier-CS's checker and 70
# test cases, 2 s CPU / 256 MB per case, score 0..100), copied next to it.
POLYOMINO_GRADER = '''\
"""Frontier-CS algorithmic #0 (Polyominoes), scored by the local judge.

Written by scripts/server/coral_tools.py prepare in place of CORAL's grader,
which calls the frontier_cs package and its go-judge Docker server. judge.py
is benchmarks/coral_tasks/polyominoes/judge.py, the judge every other method
in the comparison is scored by.
"""

from __future__ import annotations

import os
from pathlib import Path

from coral.grader import TaskGrader
from coral.types import ScoreBundle


class Grader(TaskGrader):
    def evaluate(self) -> ScoreBundle:
        solution = Path(self.codebase_path) / "solution.cpp"
        if not solution.exists():
            return self.score(0.0, feedback="No solution.cpp found in workspace.")
        code = solution.read_text()
        if not code.strip():
            return self.score(0.0, feedback="solution.cpp is empty.")
        os.environ["FRONTIER_CS_DIR"] = str(self.args["frontier_cs_dir"])  # read when judge is imported
        from . import judge

        r = judge.judge(code, workers=int(self.args.get("judge_workers", 3)))
        return self.score(r.score, feedback=r.feedback())
'''


def _setup_polyominoes(job: Path, cfg: dict, frontier_cs_dir: str, judge_workers: int) -> list[str]:
    """Swap in the local judge, and the statement that judge's checker implements."""
    problem = Path(frontier_cs_dir) / "algorithmic" / "problems" / "0"
    if not (problem / "chk.cc").exists():
        sys.exit(f"no Frontier-CS problem 0 at {problem} (run scripts/server/setup_coral.sh)")
    pkg = job / "grader" / "src" / "frontier_cs_algo_grader"
    (pkg / "grader.py").write_text(POLYOMINO_GRADER)
    shutil.copy2(POLYOMINO_JUDGE, pkg / "judge.py")
    shutil.copy2(problem / "statement.txt", job / "seed" / "statement.txt")
    g = cfg["grader"]
    g["setup"] = [s for s in g.get("setup", []) if "Frontier-CS" not in s]   # the frontier_cs package
    g["args"] = {"problem_id": "0", "frontier_cs_dir": str(Path(frontier_cs_dir).resolve()),
                 "judge_workers": judge_workers}
    return [
        "frontier_cs_algo_grader/grader.py: scores with the local judge (judge.py, as the baselines "
        "and SpecEvo) instead of frontier_cs's go-judge Docker server",
        f"seed/statement.txt: Frontier-CS's ({problem}), the statement its checker implements; "
        "CORAL 0123dfb ships an older one (square W=H, score 1e5*sum(k)/A)",
    ]


def cmd_prepare(a: argparse.Namespace) -> None:
    import yaml

    src = Path(a.coral_dir) / TASKS[a.task]["src"]
    job = Path(a.job_dir).resolve()
    if not (src / "task.yaml").exists():
        sys.exit(f"no CORAL task at {src}")
    job.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((src / "task.yaml").read_text())
    grader_cfg = cfg.setdefault("grader", {})
    # grader.private (hidden grader data, e.g. kernel_builder's frozen simulator)
    # is resolved next to task.yaml, so it travels with grader/ and seed/.
    for sub in ("grader", "seed", *grader_cfg.get("private", [])):
        if (job / sub).exists():
            shutil.rmtree(job / sub)
        shutil.copytree(src / sub, job / sub, ignore=shutil.ignore_patterns(".venv", "__pycache__", "*.egg-info"))
    patched = _patch_grader(a.task, job / "grader")
    if a.task == "polyominoes":
        patched += _setup_polyominoes(job, cfg, a.frontier_cs_dir, a.judge_workers)

    model_name = a.model.rsplit("/", 1)[-1]          # what OpenCode asks the gateway for
    gateway_url = f"http://localhost:{a.port}/v1"

    # OpenCode provider: OpenAI-compatible chat completions to the gateway. The
    # runtime puts the agent's gateway key in OPENAI_API_KEY, so the gateway
    # attributes every call to its agent.
    opencode = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "coral": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "CORAL gateway",
                "options": {"baseURL": gateway_url, "apiKey": "{env:OPENAI_API_KEY}", "includeUsage": True},
                "models": {model_name: {"name": model_name, "limit": {"context": a.context, "output": a.max_output}}},
            }
        },
    }
    (job / "seed" / "opencode.json").write_text(json.dumps(opencode, indent=2) + "\n")

    litellm = {
        "model_list": [{
            "model_name": model_name,
            "litellm_params": {"model": a.model, "api_key": "os.environ/OPENROUTER_API_KEY"},
        }],
        "litellm_settings": {"drop_params": True},
    }
    (job / "litellm_config.yaml").write_text(yaml.safe_dump(litellm, sort_keys=False))

    # A preset (kernel_builder: multi-island) is dropped with the agents and
    # islands it brings; the grader install it also carried is kept.
    if cfg.pop("preset", None) and not grader_cfg.get("setup"):
        grader_cfg["setup"] = ["uv pip install -e ./grader"]
    cfg.pop("islands", None)
    cfg["agents"] = {
        "count": a.agents,
        "runtime": "opencode",
        "model": f"coral/{model_name}",
        "gateway": {"enabled": True, "port": a.port, "config": "./litellm_config.yaml"},
    }
    ws = cfg.setdefault("workspace", {})
    ws["results_dir"] = str(job / "results")
    ws["repo_path"] = str(job / "seed")
    cfg["run"] = {"session": "local", "verbose": False, "ui": False}
    header = (f"# Written by scripts/server/coral_tools.py prepare from CORAL {TASKS[a.task]['src']}\n"
              f"# ({a.agents} OpenCode agent(s), {a.model} via the CORAL gateway on :{a.port}).\n"
              + "".join(f"# grader patch: {p}\n" for p in patched))
    (job / "task.yaml").write_text(header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=1000))
    print(f"prepared {job}")


# ---------------------------------------------------------------------------
# reading a job
# ---------------------------------------------------------------------------


def _coral_dir(job: Path) -> Path | None:
    runs = sorted(glob.glob(str(job / "results" / "*" / "*" / ".coral")), key=os.path.getmtime)
    return Path(runs[-1]) if runs else None


def _pid_alive(job: Path) -> bool:
    try:
        pid = int((job / ".run.pid").read_text().strip())
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _read_float(p: Path) -> float | None:
    try:
        return float(p.read_text().strip())
    except Exception:
        return None


def _prices(model: str, group_dir: Path | None) -> dict:
    if group_dir is not None:
        try:
            table = json.loads((group_dir / "pricing.json").read_text())
            if model in table:
                return table[model]
        except Exception:
            pass
    return DEFAULT_PRICES.get(model, DEFAULT_PRICES["openrouter/openai/gpt-5"])


def _num(d: dict | None, *keys) -> float:
    if not isinstance(d, dict):
        return 0.0
    for k in keys:
        v = d.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0


def _usage_of(entry: dict) -> dict | None:
    resp = entry.get("response")
    if not isinstance(resp, dict):
        return None
    u = resp.get("usage")
    if not isinstance(u, dict) and isinstance(resp.get("response"), dict):   # Responses API envelope
        u = resp["response"].get("usage")
    if not isinstance(u, dict):
        return None
    pin = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
    pout = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    return {
        "input": _num(u, "prompt_tokens", "input_tokens"),
        "output": _num(u, "completion_tokens", "output_tokens"),
        "cached": _num(pin, "cached_tokens") + _num(u, "cache_read_input_tokens"),
        "reasoning": _num(pout, "reasoning_tokens"),
        "cost": u.get("cost") if isinstance(u.get("cost"), (int, float)) else None,
    }


def _api_key() -> str | None:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env = Path(__file__).resolve().parents[2] / ".env"
    try:
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() in ("OPENROUTER_API_KEY", "OPENAI_API_KEY") and v.strip().startswith("sk-or-"):
                return v.strip()
    except OSError:
        pass
    return None


def fetch_generation(gen_id: str) -> dict | None:
    """Tokens and exact cost of one OpenRouter call, from its generation id."""
    import urllib.request

    key = _api_key()
    if not key:
        return None
    req = urllib.request.Request(f"https://openrouter.ai/api/v1/generation?id={gen_id}",
                                 headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            d = json.load(resp).get("data") or {}
    except Exception:
        return None
    if "total_cost" not in d:
        return None
    return {
        "input": float(d.get("native_tokens_prompt") or d.get("tokens_prompt") or 0),
        "output": float(d.get("native_tokens_completion") or d.get("tokens_completion") or 0),
        "cached": float(d.get("native_tokens_cached") or 0),
        "reasoning": float(d.get("native_tokens_reasoning") or 0),
        "cost": float(d.get("total_cost") or 0),
    }


def gateway_usage(coral: Path, job: Path, model: str, group_dir: Path | None) -> dict:
    """Totals over the gateway log, resumed from a cached byte offset."""
    log = coral / "public" / "gateway" / "requests.jsonl"
    cache_path = job / ".usage_cache.json"
    zero = {"log": str(log), "offset": 0, "calls": 0, "errors": 0, "no_usage": 0, "input": 0.0, "output": 0.0,
            "cached": 0.0, "reasoning": 0.0, "cost": 0.0, "cost_from_provider": 0, "latency_ms": 0.0}
    try:
        tot = json.loads(cache_path.read_text())
        if tot.get("log") != str(log):
            tot = dict(zero)
    except Exception:
        tot = dict(zero)
    if not log.exists():
        return tot
    price = _prices(model, group_dir)
    with log.open("rb") as fh:
        fh.seek(tot["offset"])
        while True:
            line = fh.readline()
            if not line or not line.endswith(b"\n"):
                break                                   # a half-written last line: read it next time
            tot["offset"] = fh.tell()
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "chat/completions" not in str(e.get("path", "")) and "responses" not in str(e.get("path", "")) \
                    and "messages" not in str(e.get("path", "")):
                continue
            if int(e.get("status_code") or 0) != 200:
                tot["errors"] += 1
                continue
            tot["calls"] += 1
            tot["latency_ms"] += float(e.get("duration_ms") or 0)
            u = _usage_of(e)
            if u is None:
                # A stream cut before its usage chunk (e.g. the agent stopped
                # mid-answer). OpenRouter still knows the call by its id.
                rid = e["response"].get("id") if isinstance(e.get("response"), dict) else None
                if isinstance(rid, str) and rid.startswith("gen-"):
                    tot.setdefault("pending", []).append(rid)
                else:
                    tot["no_usage"] += 1
                continue
            for k in ("input", "output", "cached", "reasoning"):
                tot[k] += u[k]
            if u["cost"] is not None:
                tot["cost"] += u["cost"]
                tot["cost_from_provider"] += 1
            else:
                uncached = max(0.0, u["input"] - u["cached"])
                tot["cost"] += (uncached * price.get("prompt", 0) + u["cached"] * price.get("input_cache_read", price.get("prompt", 0))
                                + u["output"] * price.get("completion", 0))
    still = []
    for rid in tot.get("pending", []):
        g = fetch_generation(rid)
        if g is None:
            still.append(rid)          # not available yet / network: retry on the next status
            continue
        tot["input"] += g["input"]
        tot["output"] += g["output"]
        tot["cached"] += g["cached"]
        tot["reasoning"] += g["reasoning"]
        tot["cost"] += g["cost"]
        tot["cost_from_provider"] += 1
        tot["from_generation_api"] = tot.get("from_generation_api", 0) + 1
    tot["pending"] = still
    try:
        cache_path.write_text(json.dumps(tot))
    except OSError:
        pass
    return tot


def attempts(coral: Path) -> list[dict]:
    out = []
    for f in (coral / "public" / "attempts").glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    return out


def graded(att: list[dict]) -> list[dict]:
    """Attempts CORAL's grader has finished with (crashes and timeouts included): the "evals"."""
    return [x for x in att if x.get("status") != "pending"]


def count_graded(job: Path) -> int:
    coral = _coral_dir(job)
    return len(graded(attempts(coral))) if coral else 0


def collect(lane: str, task: str, job: Path, model: str, group_dir: Path | None) -> dict:
    r: dict = {"lane": lane, "task": task, "dir": str(job), "state": "PENDING"}
    if not job.exists() or not (job / ".started_at").exists():
        return r
    started = _read_float(job / ".started_at")
    ended = _read_float(job / ".ended_at")
    running = _pid_alive(job)
    rc = (job / ".exit_code").read_text().strip() if (job / ".exit_code").exists() else None
    r["wall_s"] = ((time.time() if running or not ended else ended) - started) if started else None
    coral = _coral_dir(job)
    att = attempts(coral) if coral else []
    # `coral eval --tune` attempts are sweeps, not submissions (kernel_builder's
    # run a smaller workload, so their cycles are not comparable): CORAL leaves
    # them off its leaderboard, and so does "best". They still count as evals.
    tune = [x for x in att if (x.get("metadata") or {}).get("budget_class") == "tune"]
    final = [x for x in att if x.get("status") != "pending" and x.get("score") is not None and x not in tune]
    capped = (job / ".eval_cap_reached").exists()
    if running:
        r["state"] = "RUNNING"
    elif (job / ".stopped_by_user").exists():
        r["state"] = "STOPPED"
    # 124: stopped by the wall-clock budget; the eval cap stops CORAL with SIGTERM
    # (exit 0, or 143 if it dies of the signal). Both are the normal end.
    elif rc in ("0", "124") or (capped and rc is not None):
        r["state"] = "DONE" if final else "NO_EVALS"
        r["ended_by"] = "evals" if capped else ("time" if rc == "124" else "coral")
    elif rc is not None:
        r["state"] = f"FAILED({rc})"
    else:
        r["state"] = "STOPPED"
    spec = TASKS.get(task, {"direction": "maximize", "best": lambda s: s})
    pick = min if spec["direction"] == "minimize" else max
    best = pick(final, key=lambda x: float(x["score"]), default=None)
    r["attempts"] = len(graded(att))
    r["tune_attempts"] = len(tune)
    r["crashed"] = len([x for x in att if x.get("status") in ("crashed", "timeout")])
    if best is not None:
        r["best_score"] = float(best["score"])          # CORAL's own number (ratio / score / cycles)
        r["best"] = spec["best"](r["best_score"])       # the comparison's number
        r["best_title"] = str(best.get("title", ""))[:80]
        r["best_at"] = best.get("timestamp")
    if coral is not None:
        u = gateway_usage(coral, job, model, group_dir)
        r.update({
            "llm_calls": u["calls"], "llm_errors": u["errors"],
            "calls_without_usage": u["no_usage"] + len(u.get("pending", [])),
            "input_tokens": int(u["input"]), "cached_tokens": int(u["cached"]),
            "output_tokens": int(u["output"]), "reasoning_tokens": int(u["reasoning"]),
            "cost": u["cost"], "cost_from_provider": u["cost_from_provider"],
            "s_per_call": (u["latency_ms"] / 1000 / u["calls"]) if u["calls"] else None,
            "run_dir": str(coral.parent),
        })
    return r


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------


def _f(v, fmt="{:.4f}"):
    return "-" if v is None else fmt.format(v)


def _dur(s):
    if s is None:
        return "-"
    s = int(s)
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _k(v):
    if v is None:
        return "-"
    return f"{v / 1e6:.2f}M" if v >= 1e6 else f"{v / 1e3:.0f}k"


BEST_LEGEND = {
    "circle_packing": "best = sum of radii (coral = CORAL's ratio to the best known 2.635977)",
    "circle_packing_rect": "best = sum of radii (coral = CORAL's ratio to the best known 2.365832)",
    "polyominoes": "best = coral = score 0..100 from the local judge (as run_pk_queue.sh)",
    "kernel_builder": "best = 147734 / cycles (coral = cycles, lower is better; as run_pk_queue.sh)",
}


def _evals(r: dict, cap: int) -> str:
    n = r.get("attempts")
    if n is None:
        return "-"
    return f"{n}/{cap}" if cap else str(n)


def read_jobs() -> list[tuple[str, str, Path]]:
    jobs = []
    for line in sys.stdin:
        parts = line.split()
        if len(parts) == 3:
            jobs.append((parts[0], parts[1], Path(parts[2])))
    return jobs


def cmd_status(a: argparse.Namespace) -> None:
    group = Path(a.group_dir) if a.group_dir else None
    cap = getattr(a, "max_evals", 0) or 0
    hdr = (f"{'lane':<4} {'task':<20} {'state':<9} {'best':>9} {'coral':>8} {'evals':>7} {'crash':>5} "
           f"{'cost':>7} {'calls':>5} {'in tok':>7} {'cached':>7} {'out tok':>7} {'reason':>7} {'s/call':>6} {'wall':>6}")
    print(hdr)
    print("-" * len(hdr))
    rows = [collect(l, t, d, a.model, group) for l, t, d in read_jobs()]
    for r in rows:
        raw = "{:.0f}" if r["task"] == "kernel_builder" else "{:.4f}"
        print(f"{r['lane']:<4} {r['task']:<20} {r['state']:<9} {_f(r.get('best'), '{:.6f}'):>9} "
              f"{_f(r.get('best_score'), raw):>8} {_evals(r, cap):>7} {r.get('crashed', '-')!s:>5} "
              f"{_f(r.get('cost'), '${:.2f}'):>7} {r.get('llm_calls', '-')!s:>5} {_k(r.get('input_tokens')):>7} "
              f"{_k(r.get('cached_tokens')):>7} {_k(r.get('output_tokens')):>7} {_k(r.get('reasoning_tokens')):>7} "
              f"{_f(r.get('s_per_call'), '{:.1f}'):>6} {_dur(r.get('wall_s')):>6}")
    for r in rows:
        if r.get("ended_by"):
            why = {"evals": "stopped at the eval cap" + (f" ({cap})" if cap else ""),
                   "time": "ran out its clock", "coral": "CORAL ended on its own"}[r["ended_by"]]
            print(f"note: {r['lane']} {r['task']}: {why}")
        if r.get("calls_without_usage"):
            print(f"note: {r['lane']} {r['task']}: {r['calls_without_usage']} LLM call(s) logged without token "
                  f"usage -- their cost is missing from the total")
        if r.get("tune_attempts"):
            print(f"note: {r['lane']} {r['task']}: {r['tune_attempts']} of the evals were `coral eval --tune` "
                  f"sweeps, left out of best")
    print()
    for t in dict.fromkeys(r["task"] for r in rows):
        print(f"{t}: {BEST_LEGEND.get(t, 'best = coral = CORAL score')}")
    print("evals = graded attempts" + (f" (of the {cap} cap)" if cap else "")
          + "; cost from token usage × OpenRouter prices")


def cmd_compare(a: argparse.Namespace) -> None:
    group = Path(a.group_dir) if a.group_dir else None
    rows = [collect(l, t, d, a.model, group) for l, t, d in read_jobs()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["lane", "task", "state", "ended_by", "best", "best_score", "attempts", "crashed", "cost", "llm_calls",
            "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "s_per_call", "wall_s", "run_dir"]
    with (out / "comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2, default=str))
    md = ["| task | state | best | CORAL score | evals | cost | LLM calls | input tok | cached | output tok | reasoning | s/call | wall |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['task']} | {r['state']} | {_f(r.get('best'), '{:.6f}')} | {_f(r.get('best_score'))} | "
                  f"{r.get('attempts', '-')} | {_f(r.get('cost'), '${:.2f}')} | {r.get('llm_calls', '-')} | "
                  f"{_k(r.get('input_tokens'))} | {_k(r.get('cached_tokens'))} | {_k(r.get('output_tokens'))} | "
                  f"{_k(r.get('reasoning_tokens'))} | {_f(r.get('s_per_call'), '{:.1f}')} | {_dur(r.get('wall_s'))} |")
    (out / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}/comparison.{{md,csv,json}}")


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--task", required=True, choices=sorted(TASKS))
    pp.add_argument("--job-dir", required=True)
    pp.add_argument("--coral-dir", required=True)
    pp.add_argument("--port", type=int, required=True)
    pp.add_argument("--model", required=True)
    pp.add_argument("--agents", type=int, default=1)
    pp.add_argument("--context", type=int, default=400000)
    pp.add_argument("--max-output", type=int, default=128000)
    pp.add_argument("--frontier-cs-dir", default=os.environ.get("FRONTIER_CS_DIR", DEFAULT_FRONTIER_CS_DIR),
                    help="polyominoes: Frontier-CS checkout with problem 0's checker and test cases")
    pp.add_argument("--judge-workers", type=int, default=3, help="polyominoes: test cases judged in parallel")
    for name in ("status", "compare"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", default="openrouter/openai/gpt-5")
        sp.add_argument("--group-dir", default=None)
        if name == "compare":
            sp.add_argument("--out", required=True)
        else:
            sp.add_argument("--max-evals", type=int, default=0, help="show evals as n/N")
    cp = sub.add_parser("count")
    cp.add_argument("--job-dir", required=True)
    a = p.parse_args()
    if a.cmd == "count":
        print(count_graded(Path(a.job_dir)))
        return 0
    {"prepare": cmd_prepare, "status": cmd_status, "compare": cmd_compare}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
