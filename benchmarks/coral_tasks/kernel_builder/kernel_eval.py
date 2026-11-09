"""Scoring for CORAL's "Kernel Engineering" task (Anthropic's performance take-home).

A candidate defines ``KernelBuilder`` whose ``build_kernel(forest_height,
n_nodes, batch_size, rounds)`` emits VLIW SIMD instructions for a frozen
simulator (``frozen_problem.py``, copied unchanged from CORAL
examples/kernel_builder/taskdata). The workload mirrors CORAL's grader
(kernel_builder_grader.grader, real mode): forest_height=10, rounds=16,
batch_size=256, 8 correctness runs on fresh random trees, cycle count of the
last run. Any wrong output scores 0.

Two steps, two processes, so candidate code can never touch the simulator:
  1. build  -- in the candidate's process: run build_kernel once and turn the
     program into plain JSON (engines -> slots of ints/strings);
  2. judge  -- in a fresh interpreter that imports only this module and
     frozen_problem: rebuild the instruction tuples from that JSON and run
     the correctness/timing loop.
(CORAL instead execs the candidate next to the simulator, where it could
monkeypatch Machine or the reference. build_kernel never sees the data, so
building once equals CORAL's rebuild-per-run for any deterministic builder.)

Score = BASELINE_CYCLES / cycles, the speed-up over the naive scalar kernel
(higher is better): 1.0 at 147,734 cycles, 108.4 at the official best 1,363,
133.9 at CORAL's 4-agent 1,103. ``cycles`` is reported alongside.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE_CYCLES = 147_734
BEST_KNOWN_CYCLES = 1_363
REAL_PARAMS = {"forest_height": 10, "rounds": 16, "batch_size": 256, "iterations": 8}
JUDGE_TIMEOUT_S = 300.0


def score_of(cycles: int) -> float:
    return BASELINE_CYCLES / cycles if cycles > 0 else 0.0


# ---------------------------------------------------------------------------
# 1. build (untrusted side)
# ---------------------------------------------------------------------------


def _plain(x):
    """Instruction data as JSON: tuples/lists of ints and strings only."""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, str)):
        return x
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if hasattr(x, "__index__"):           # numpy ints and the like
        return int(x)
    raise TypeError(f"instruction operand of unsupported type {type(x).__name__}: {x!r}")


def build_program(KernelBuilder, params: dict = REAL_PARAMS) -> dict:
    h = params["forest_height"]
    kb = KernelBuilder()
    kb.build_kernel(h, 2 ** (h + 1) - 1, params["batch_size"], params["rounds"])
    instrs = []
    for ins in kb.instrs:
        if not isinstance(ins, dict):
            raise TypeError(f"each instruction must be a dict of engine -> slots, got {type(ins).__name__}")
        instrs.append({str(eng): [_plain(s) for s in slots] for eng, slots in ins.items()})
    smap = {}
    try:
        for addr, (name, length) in (kb.debug_info().scratch_map or {}).items():
            smap[str(int(addr))] = [str(name), int(length)]
    except Exception:
        smap = {}                          # debug info is optional for scoring
    return {"instrs": instrs, "scratch_map": smap}


# ---------------------------------------------------------------------------
# 2. judge (trusted side)
# ---------------------------------------------------------------------------


def _tuples(x):
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def simulate(program: dict, params: dict = REAL_PARAMS) -> dict:
    """Correctness/timing loop on a JSON program. Run only in a clean process."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from frozen_problem import N_CORES, DebugInfo, Input, Machine, Tree, build_mem_image, reference_kernel2

    instrs = [{eng: [_tuples(s) for s in slots] for eng, slots in ins.items()} for ins in program["instrs"]]
    dbg = DebugInfo(scratch_map={int(k): tuple(v) for k, v in program.get("scratch_map", {}).items()})
    cycles = None
    for i in range(params["iterations"]):
        forest = Tree.generate(params["forest_height"])
        inp = Input.generate(forest, params["batch_size"], params["rounds"])
        mem = build_mem_image(forest, inp)

        machine = Machine(mem, instrs, dbg, n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        machine.run()

        for ref_mem in reference_kernel2(mem):
            pass
        values_p = ref_mem[6]
        actual = machine.mem[values_p: values_p + len(inp.values)]
        expected = ref_mem[values_p: values_p + len(inp.values)]
        cycles = int(machine.cycle)
        if actual != expected:
            return {"is_correct": False, "cycles": cycles,
                    "error_msg": f"iteration {i}: incorrect output values"}
    return {"is_correct": True, "cycles": cycles, "error_msg": ""}


def _last_json(text: str) -> dict | None:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return None


def judge_program(program: dict, timeout: float = JUDGE_TIMEOUT_S) -> dict:
    """Run ``simulate`` on ``program`` in a fresh interpreter (``python -I``)."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(program, fh)
        path = fh.name
    try:
        script = (
            "import json, sys\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import kernel_eval\n"
            f"print(json.dumps(kernel_eval.simulate(json.load(open({path!r})))))\n"
        )
        proc = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True,
                              timeout=timeout, cwd=str(HERE))
    except subprocess.TimeoutExpired:
        return {"error": f"simulation timed out after {timeout:.0f}s"}
    finally:
        os.unlink(path)
    res = _last_json(proc.stdout)
    if res is None:
        return {"error": f"simulator exit {proc.returncode}: {(proc.stderr or '').strip()[-1500:]}"}
    return res


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def run_builder(KernelBuilder, params: dict = REAL_PARAMS) -> dict:
    """SpecEvo: the class lives in this (candidate) process; judge elsewhere."""
    try:
        program = build_program(KernelBuilder, params)
    except Exception as e:
        return {"error": f"build_kernel failed: {type(e).__name__}: {e}"[:2000]}
    return judge_program(program)


def run_program_file(program_path: str, timeout: float = JUDGE_TIMEOUT_S) -> dict:
    """Baselines: build in one interpreter (exec of the candidate file), judge in another."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        out = fh.name
    try:
        script = (
            "import json, sys, traceback\n"
            f"sys.path.insert(0, {str(HERE)!r})\n"
            "import kernel_eval\n"
            f"src = open({os.path.abspath(program_path)!r}).read()\n"
            "ns = {'__name__': '__main__'}\n"
            "exec(src, ns)\n"
            "if 'KernelBuilder' not in ns:\n"
            "    raise SystemExit('KernelBuilder class not found in program')\n"
            "prog = kernel_eval.build_program(ns['KernelBuilder'])\n"
            f"json.dump(prog, open({out!r}, 'w'))\n"
        )
        try:
            proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"error": f"building the kernel timed out after {timeout:.0f}s"}
        if proc.returncode != 0 or os.path.getsize(out) == 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            return {"error": f"exit {proc.returncode}: {tail}" if tail else f"exit {proc.returncode}, no program"}
        with open(out) as fh:
            program = json.load(fh)
    finally:
        os.unlink(out)
    return judge_program(program, timeout=timeout)


def feedback(res: dict) -> str:
    if "error" in res:
        return f"Error: {res['error']}"
    if not res.get("is_correct"):
        return f"Kernel produces incorrect output: {res.get('error_msg', '')}"
    c = res["cycles"]
    return (f"Cycles: {c:,} | Speedup: {score_of(c):.2f}x | Baseline: {BASELINE_CYCLES:,} | "
            f"Best known: {BEST_KNOWN_CYCLES:,}" + (" | NEW RECORD!" if c <= BEST_KNOWN_CYCLES else ""))
