"""Shared CO-Bench evaluation engine for SkyDiscover.

This module adapts the CO-Bench benchmark (Sun et al., 2025 —
https://github.com/sunnweiwei/CO-Bench) so a single ``solve`` function can be
scored the same way for BOTH integration paths in this repo:

  * the *baseline* runners (openevolve/gepa/adaevolve/evox) via
    ``benchmarks/co_bench/<slug>/evaluator.py``; and
  * the *BLADE / LEVI* runner via ``levi/examples/co_bench/<slug>/problem.py``.

Each CO-Bench task ships a ``config.py`` (vendored under ``data/<task>/``)
exporting ``load_data``, ``eval_func``, ``DESCRIPTION`` and a ``solve``
template, plus optionally ``norm_score`` and ``get_dev``.  We evaluate a
candidate ``solve`` on the task's test-case files exactly as CO-Bench does:
each *instance* runs in isolation under a hard per-instance time limit (10s in
the paper), raw scores are normalised against the best-known objective, and the
per-problem score is the mean over instances (a program error, constraint
violation or timeout scores 0 for that instance).

Two per-instance execution strategies keep the semantics identical across both
paths while respecting their process models:

  * **fork subprocess** (default, non-daemon caller e.g. the baseline
    evaluator): hard-kills a runaway/looping ``solve`` — matches CO-Bench.
  * **in-process SIGALRM** (daemon caller e.g. a LEVI worker, which forbids
    child processes): interrupts pure-Python loops in the worker's main thread.

Runtime knobs (env vars, all optional):
  COBENCH_TIMEOUT        per-instance time limit in seconds (default 10)
  COBENCH_MAX_CASES      cap on number of test-case files evaluated
  COBENCH_MAX_INSTANCES  cap on instances per test-case file
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

# Directory holding the vendored CO-Bench tasks: data/<task>/config.py + files.
DATA_ROOT = Path(__file__).resolve().parent / "data"

# Canonical slug -> CO-Bench task name (one representative per CO-Bench category).
TASKS: dict[str, str] = {
    "bin_packing_1d": "Bin packing - one-dimensional",          # Packing
    "non_guillotine_cutting": "Constrained non-guillotine cutting",  # Cutting
    "warehouse_location_uncap": "Uncapacitated warehouse location",  # Facility location
    "flow_shop": "Flow shop scheduling",                        # Scheduling
    "tsp": "Travelling salesman problem",                       # Routing
    "gap": "Generalised assignment problem",                    # Assignment
    "steiner": "Euclidean Steiner problem",                     # Tree
    "graph_coloring": "Graph colouring",                        # Graph & set
}

CATEGORY: dict[str, str] = {
    "bin_packing_1d": "Packing",
    "non_guillotine_cutting": "Cutting",
    "warehouse_location_uncap": "Facility location",
    "flow_shop": "Scheduling",
    "tsp": "Routing",
    "gap": "Assignment",
    "steiner": "Tree",
    "graph_coloring": "Graph & set",
}


def _env_int(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def default_timeout() -> float:
    raw = os.environ.get("COBENCH_TIMEOUT", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 10.0


# --------------------------------------------------------------------------- #
# CO-Bench scoring helpers (vendored verbatim from evaluation/utils.py so we do
# not depend on the CO-Bench package, which pulls litellm/openai/etc.).
# --------------------------------------------------------------------------- #
def _average_score(results: dict) -> float:
    if not results:
        return 0.0
    return sum(
        (sum(x if not isinstance(x, str) else 0 for x in scores) / len(scores)
         if not error_message else 0)
        for scores, error_message in
        (results.get(case, (None, "No result")) for case in results.keys())
    ) / len(results)


def _filter_dev(results: dict, dev: Optional[dict]) -> dict:
    if dev is None:
        return results
    out = {}
    for case, (scores, error_message) in results.items():
        if case not in dev:
            continue
        dev_list = dev[case] or [0]
        sel = [s for idx, s in enumerate(scores) if idx in dev_list]
        if sel:
            out[case] = (sel, error_message)
    return out


def _filter_test(results: dict, dev: Optional[dict]) -> dict:
    if dev is None:
        return results
    out = {}
    for case, (scores, error_message) in results.items():
        if case not in dev:
            out[case] = (scores, error_message)
            continue
        dev_list = dev[case] or [0]
        sel = [s for idx, s in enumerate(scores) if idx not in dev_list]
        if sel:
            out[case] = (sel, error_message)
    return out


def _valid_rate(results: dict) -> float:
    total = numeric = 0
    for scores, err in results.values():
        if err or scores is None:
            total += 1
            continue
        for s in scores:
            total += 1
            if not isinstance(s, str):
                numeric += 1
    return numeric / total if total else 0.0


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #
def load_config(task: str, data_root: Path = DATA_ROOT):
    """Import a task's vendored config.py and return the module."""
    config_path = data_root / task / "config.py"
    if not config_path.is_file():
        raise FileNotFoundError(f"CO-Bench config not found: {config_path}")
    spec = importlib.util.spec_from_file_location(
        f"cobench_cfg_{abs(hash(str(config_path)))}", str(config_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod, str(config_path)


def problem_description(task: str, data_root: Path = DATA_ROOT) -> str:
    """The CO-Bench problem description + solve template (agent-facing prompt)."""
    import ast

    config_path = data_root / task / "config.py"
    src = config_path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    solve_src = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            solve_src = "\n".join(lines[node.lineno - 1: node.end_lineno])
            break
    mod, _ = load_config(task, data_root)
    return f"{mod.DESCRIPTION}\n\n# Implement in Solve Function\n\n{solve_src}"


def solve_template(task: str, data_root: Path = DATA_ROOT) -> str:
    import ast

    config_path = data_root / task / "config.py"
    src = config_path.read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "solve":
            return "\n".join(lines[node.lineno - 1: node.end_lineno])
    raise ValueError(f"No solve() found in {config_path}")


def list_test_cases(task: str, data_root: Path = DATA_ROOT) -> list[str]:
    d = data_root / task
    return sorted(
        f.name for f in d.iterdir()
        if f.is_file() and not f.name.endswith(".py") and f.name != "__pycache__"
    )


# --------------------------------------------------------------------------- #
# Per-instance execution
# --------------------------------------------------------------------------- #
def _subproc_target(solve_source, config_path, instance, q):
    try:
        ns: dict[str, Any] = {}
        exec(solve_source, ns)
        solve_fn = ns.get("solve")
        if not callable(solve_fn):
            q.put("Exception: candidate does not define solve()")
            return
        spec = importlib.util.spec_from_file_location("cobench_child_cfg", config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        solution = solve_fn(**instance)
        solution = {str(k): v for k, v in solution.items()}
        score = cfg.eval_func(**instance, **solution)
        q.put(score)
    except Exception as e:  # noqa: BLE001
        q.put(f"Exception: {e}")


def _run_instance_subprocess(solve_source, config_path, instance, timeout) -> Any:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_subproc_target, args=(solve_source, config_path, instance, q))
    p.start()
    p.join(timeout + 1)
    if p.is_alive():
        p.terminate()
        p.join(1)
        if p.is_alive():
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
        return f"Timeout ({timeout}s)"
    try:
        return q.get_nowait()
    except Exception:  # noqa: BLE001
        return "No result"


class _AlarmTimeout(Exception):
    pass


def _run_instance_sigalrm(solve_fn, eval_func, instance, timeout) -> Any:
    def _handler(signum, frame):
        raise _AlarmTimeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        solution = solve_fn(**instance)
        solution = {str(k): v for k, v in solution.items()}
        return eval_func(**instance, **solution)
    except _AlarmTimeout:
        return f"Timeout ({timeout}s)"
    except Exception as e:  # noqa: BLE001
        return f"Exception: {e}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _run_instance_thread(solve_fn, eval_func, instance, timeout) -> Any:
    """Soft-timeout fallback (daemon + non-main-thread): cannot hard-kill."""
    box: dict[str, Any] = {}

    def _work():
        try:
            solution = solve_fn(**instance)
            solution = {str(k): v for k, v in solution.items()}
            box["r"] = eval_func(**instance, **solution)
        except Exception as e:  # noqa: BLE001
            box["r"] = f"Exception: {e}"

    t = threading.Thread(target=_work, daemon=True)
    t.start()
    t.join(timeout + 1)
    if t.is_alive():
        return f"Timeout ({timeout}s)"
    return box.get("r", "No result")


# --------------------------------------------------------------------------- #
# Top-level evaluation
# --------------------------------------------------------------------------- #
def evaluate_source(
    task: str,
    solve_source: str,
    *,
    data_root: Path = DATA_ROOT,
    timeout: Optional[float] = None,
    max_cases: Optional[int] = None,
    max_instances: Optional[int] = None,
) -> dict[str, Any]:
    """Score a candidate ``solve`` (given as source text) on a CO-Bench task.

    Returns a metric dict:
      score / dev_score / test_score / overall_score  (higher is better, 1.0 = best-known)
      valid_rate, num_cases, num_instances, feedback
    ``score`` == ``overall_score`` (mean over every instance actually evaluated) —
    this is the robust signal the search optimises, well-defined under any cap.
    ``dev_score`` / ``test_score`` reproduce CO-Bench's dev/test split and are only
    meaningful when the FULL set is run (MAX_CASES=MAX_INSTANCES=0); under a small
    cap the split can be tiny or empty because get_dev() indexes fixed positions.
    """
    if timeout is None:
        timeout = default_timeout()
    if max_cases is None:
        max_cases = _env_int("COBENCH_MAX_CASES", None)
    if max_instances is None:
        max_instances = _env_int("COBENCH_MAX_INSTANCES", None)

    cfg, config_path = load_config(task, data_root)
    cases = list_test_cases(task, data_root)
    if max_cases is not None:
        cases = cases[:max_cases]

    norm_score = getattr(cfg, "norm_score", None)
    get_dev = getattr(cfg, "get_dev", None)

    # Decide per-instance execution strategy once.
    daemon = mp.current_process().daemon
    is_main_thread = threading.current_thread() is threading.main_thread()
    use_subprocess = not daemon

    # In-process paths need a compiled solve callable.
    solve_fn = None
    if not use_subprocess:
        ns: dict[str, Any] = {}
        try:
            exec(solve_source, ns)
            solve_fn = ns.get("solve")
        except Exception as e:  # noqa: BLE001
            return _error_result(f"Candidate failed to compile: {e}")
        if not callable(solve_fn):
            return _error_result("Candidate does not define solve()")

    # Load every case's instances up front (preserving order for norm_score).
    results: dict[str, tuple] = {}
    loaded: dict[str, list] = {}
    for case in cases:
        file_path = str(data_root / task / case)
        try:
            instances = cfg.load_data(file_path)
        except Exception as e:  # noqa: BLE001
            results[case] = (None, f"load_data error: {e}")
            continue
        if max_instances is not None:
            instances = instances[:max_instances]
        loaded[case] = list(instances)

    score_arrays = {case: [None] * len(insts) for case, insts in loaded.items()}
    tasks = [(case, idx, inst)
             for case, insts in loaded.items()
             for idx, inst in enumerate(insts)]
    total_instances = len(tasks)

    if use_subprocess:
        # Baseline path: evaluate instances in PARALLEL (each in its own forked
        # subprocess with a hard timeout). This bounds a slow/looping candidate's
        # wall time to ~one timeout batch instead of instances x timeout — matching
        # CO-Bench's ParallelRun. Order is preserved via the (case, idx) key.
        workers = _env_int("COBENCH_WORKERS", None) or (os.cpu_count() or 4)
        workers = max(1, min(workers, len(tasks) or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {
                ex.submit(_run_instance_subprocess, solve_source, config_path, inst, timeout):
                    (case, idx)
                for case, idx, inst in tasks
            }
            for f in concurrent.futures.as_completed(fut):
                case, idx = fut[f]
                try:
                    score_arrays[case][idx] = f.result()
                except Exception as e:  # noqa: BLE001
                    score_arrays[case][idx] = f"Exception: {e}"
    else:
        # Daemon path (BLADE worker): no child processes allowed, so evaluate
        # sequentially under SIGALRM (main thread) or a soft thread timeout.
        # The BLADE worker's own eval_timeout bounds the whole candidate.
        for case, idx, inst in tasks:
            if is_main_thread:
                s = _run_instance_sigalrm(solve_fn, cfg.eval_func, inst, timeout)
            else:
                s = _run_instance_thread(solve_fn, cfg.eval_func, inst, timeout)
            score_arrays[case][idx] = s

    for case, insts in loaded.items():
        results[case] = (score_arrays[case], None)

    # Normalise raw scores against best-known objective.
    if callable(norm_score):
        try:
            results = norm_score(results)
        except Exception:  # noqa: BLE001 — fall back to raw on a broken norm
            pass

    dev = None
    if callable(get_dev):
        try:
            dev = get_dev()
        except Exception:  # noqa: BLE001
            dev = None

    overall = _average_score(results)
    dev_score = _average_score(_filter_dev(results, dev))
    test_score = _average_score(_filter_test(results, dev))
    valid = _valid_rate(results)

    return {
        "score": overall,
        "dev_score": dev_score,
        "test_score": test_score,
        "overall_score": overall,
        "valid_rate": valid,
        "num_cases": len(results),
        "num_instances": total_instances,
        "feedback": _feedback(results, overall),
    }


def _error_result(msg: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "dev_score": 0.0,
        "test_score": 0.0,
        "overall_score": 0.0,
        "valid_rate": 0.0,
        "num_cases": 0,
        "num_instances": 0,
        "feedback": msg,
        "error": msg,
    }


def _feedback(results: dict, avg: float, feedback_length: int = 16) -> str:
    lines = []
    for case, (scores, err) in list(results.items())[:feedback_length]:
        if err:
            lines.append(f"{case} -> Caught Error: {err}")
        elif scores is None:
            lines.append(f"{case} -> No result")
        else:
            shown = [s if isinstance(s, str) else f"{float(s):.3f}" for s in scores][:feedback_length]
            lines.append(f"{case} -> Scores: {shown}")
    lines.append(f"Avg Score {avg}")
    return "\n".join(lines)


def read_solve_source(program_path: str) -> str:
    """Return candidate source for the baseline path (whole file defines solve)."""
    return Path(program_path).read_text()
