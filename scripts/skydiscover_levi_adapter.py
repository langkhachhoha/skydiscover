"""Generic adapter that lets LEVI evolve a SkyDiscover benchmark directly.

A SkyDiscover benchmark directory looks like::

    <bench_dir>/
        initial_program.py     # seed code, may contain `EVOLVE-BLOCK-START/END`
        evaluator.py           # exposes `evaluate(program_path) -> {"combined_score": ..., ...}`
        config.yaml            # has `prompt.system_message` (problem description)

LEVI's API expects a `score_fn(fn)` (callable, not file path), a
`function_signature` string, and a `seed_program` string. This module bridges
the gap:

* :func:`load_benchmark` reads the three files and detects the entry-point
  function name + signature from the seed.
* :func:`make_score_fn` returns a *picklable* score function (a
  ``functools.partial`` of the module-level :func:`_skydiscover_score_fn`) that
  LEVI's process pool can ship to subprocess workers.
* Inside the worker, :func:`_skydiscover_score_fn` recovers the *full* evolved
  source via ``fn.__globals__["__source_code__"]`` (LEVI's
  ``evaluate_code`` helper stuffs the source in there), writes it to a temp
  file inside the benchmark dir so sibling imports resolve, then calls
  ``evaluator.evaluate(tmp_path)`` and normalizes ``combined_score → score``.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable


@dataclass
class BenchmarkSpec:
    """All the inputs LEVI needs about a SkyDiscover benchmark."""

    name: str
    bench_dir: Path
    initial_program: str         # full source of initial_program.py
    seed_program: str            # what we hand to LEVI (full file by default)
    evaluator_path: Path         # absolute path to evaluator.py
    function_name: str           # entry-point function LEVI evolves toward
    function_signature: str      # "def construct_packing():"
    problem_description: str     # extracted from config.yaml


# ---------------------------------------------------------------------------
# Benchmark discovery / parsing
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_problem_description(config_yaml: Path) -> str:
    """Pull `prompt.system_message` out of config.yaml without importing yaml.

    We don't want to add a hard dependency on PyYAML just for this. A tiny
    in-process parser is enough because the format is constrained: a top-level
    `prompt:` block with a `system_message: |-` or `system_message: '...'`.
    Falls back to "" if anything goes wrong — LEVI still works without it.
    """
    if not config_yaml.is_file():
        return ""
    try:
        import yaml  # PyYAML if available
        data = yaml.safe_load(_read_text(config_yaml)) or {}
        prompt = data.get("prompt") or {}
        msg = prompt.get("system_message")
        if isinstance(msg, str):
            return msg.strip()
    except Exception:
        pass
    return ""


def _detect_function(initial_program: str) -> tuple[str, str]:
    """Return (function_name, full_signature_line) for the function LEVI evolves.

    Strategy:
      1. If there is an `# EVOLVE-BLOCK-START` marker, look at the first
         ``def`` inside it.
      2. Otherwise, the first top-level ``def`` in the file.
    """
    src = initial_program
    start = src.find("EVOLVE-BLOCK-START")
    haystack = src[start:] if start != -1 else src

    try:
        tree = ast.parse(haystack if start == -1 else src)
    except SyntaxError as exc:
        raise ValueError(f"Cannot parse initial_program: {exc}") from exc

    fns = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not fns:
        raise ValueError("No top-level function found in initial_program.py")

    # If we have markers, prefer the first function whose line is inside them.
    if start != -1:
        lines = src.splitlines()
        block_start = src[:start].count("\n")
        end_idx = src.find("EVOLVE-BLOCK-END")
        block_end = src[:end_idx].count("\n") if end_idx != -1 else len(lines)
        in_block = [fn for fn in fns if block_start < fn.lineno - 1 < block_end]
        if in_block:
            fn = in_block[0]
        else:
            fn = fns[0]
    else:
        fn = fns[0]

    name = fn.name
    args_src = ast.unparse(fn.args)
    returns = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    signature = f"def {name}({args_src}){returns}:"
    return name, signature


def load_benchmark(bench_dir: str | Path) -> BenchmarkSpec:
    """Inspect a SkyDiscover benchmark directory and return a `BenchmarkSpec`.

    Raises ``FileNotFoundError`` if ``initial_program.py`` or ``evaluator.py``
    are missing. Docker-only benchmarks (``evaluator/`` directory with a
    Dockerfile and no plain ``evaluator.py``) are not supported here.
    """
    bench_dir = Path(bench_dir).resolve()
    initial_path = bench_dir / "initial_program.py"
    evaluator_path = bench_dir / "evaluator.py"
    config_path = bench_dir / "config.yaml"

    if not initial_path.is_file():
        raise FileNotFoundError(f"Missing initial_program.py in {bench_dir}")
    if not evaluator_path.is_file():
        raise FileNotFoundError(
            f"Missing plain Python evaluator.py in {bench_dir}. "
            "Docker-only benchmarks are not supported by this adapter."
        )

    initial_program = _read_text(initial_path)
    fn_name, fn_signature = _detect_function(initial_program)
    description = _read_problem_description(config_path)

    return BenchmarkSpec(
        name=bench_dir.name,
        bench_dir=bench_dir,
        initial_program=initial_program,
        seed_program=initial_program,
        evaluator_path=evaluator_path,
        function_name=fn_name,
        function_signature=fn_signature,
        problem_description=description or f"Evolve `{fn_name}` to maximize the evaluator's combined_score.",
    )


# ---------------------------------------------------------------------------
# Score function (runs inside the LEVI process-pool worker)
# ---------------------------------------------------------------------------

def _import_evaluator(evaluator_path: str):
    """Import ``evaluator.py`` *inside* the worker, with its dir on sys.path."""
    bench_dir = os.path.dirname(os.path.abspath(evaluator_path))
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    spec = importlib.util.spec_from_file_location(f"sd_eval_{abs(hash(evaluator_path))}", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, bench_dir


def _skydiscover_score_fn(evaluator_path: str, fn: Callable[..., Any]) -> dict[str, Any]:
    """LEVI score_fn that delegates to a SkyDiscover-style evaluator.

    Must be a top-level function so it's picklable for LEVI's process pool.
    """
    # `evaluate_code` in LEVI stores the full evolved source under this key.
    src = fn.__globals__.get("__source_code__") if hasattr(fn, "__globals__") else None
    if not src:
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            return {"error": "Could not recover source code for evolved function"}

    try:
        evaluator, bench_dir = _import_evaluator(evaluator_path)
    except Exception as exc:  # pragma: no cover - boundary
        return {"error": f"Failed to import evaluator: {exc}"}

    if not hasattr(evaluator, "evaluate"):
        return {"error": f"evaluator at {evaluator_path} has no `evaluate(program_path)`"}

    # Write the evolved code into the benchmark dir so sibling imports work.
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="_levi_candidate_", dir=bench_dir, delete=False
    )
    try:
        tmp.write(src)
        tmp.close()
        result = evaluator.evaluate(tmp.name)
    except Exception as exc:
        return {"error": f"evaluator raised: {exc}"}
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if not isinstance(result, dict):
        return {"error": f"evaluator returned {type(result).__name__}, expected dict"}

    if "error" in result and "combined_score" not in result:
        return {"error": str(result["error"])}

    score = result.get("combined_score")
    if score is None:
        score = result.get("score", 0.0)

    out: dict[str, Any] = {"score": float(score)}
    for k, v in result.items():
        if k == "combined_score":
            continue
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def make_score_fn(evaluator_path: str | Path) -> Callable[[Callable[..., Any]], dict[str, Any]]:
    """Build a picklable score_fn bound to a specific SkyDiscover evaluator."""
    return partial(_skydiscover_score_fn, str(Path(evaluator_path).resolve()))
