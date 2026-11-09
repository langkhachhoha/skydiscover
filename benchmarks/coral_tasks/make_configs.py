"""Write config.yaml for the two CORAL stress-test tasks.

    python benchmarks/coral_tasks/make_configs.py

The system prompts carry the same task text CORAL gives its agents
(task.yaml description + tips of examples/frontier_cs_algo/0 and
examples/kernel_builder at CORAL 0123dfb), minus CORAL's own workflow
instructions; the Polyominoes prompt embeds the full Frontier-CS statement,
which CORAL agents read from statement.txt. SpecEvo's problem.py files under
levi/examples/{polyominoes,kernel_builder} import the same text from here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "polyominoes"))

POLYOMINO_TASK = """\
Solve algorithmic problem #0 from the Frontier-CS benchmark: Pack the Polyominoes (Reflections Allowed).

Write a self-contained C++17 solution that:
- Reads input from stdin
- Writes output to stdout
- Compiles with `g++ -std=gnu++17 -O2`
- Runs within the time limit of 2s per test case
- Uses at most 256MB of memory

Strategy tips:
- Start with a correct brute-force, then optimize.
- Many problems reward heuristics: simulated annealing, beam search, greedy + local search.
- Check the scoring formula in the statement — some problems score relative to a baseline.

Scoring:
- Your score is 0-100 based on solution quality across 70 test cases (mean of the per-case score).
- Invalid or crashing solutions score 0 on that case. Time limit: 2s CPU per test case. Memory limit: 256MB.
"""

KERNEL_TASK = """\
Optimize a VLIW SIMD kernel for a tree traversal algorithm on a simulated machine
(Anthropic's performance-engineering take-home).

The machine has the following engines and slot limits per cycle:
- alu: 12 scalar operations
- valu: 6 vector operations (VLEN=8, processes 8 elements at once)
- load: 2 memory reads
- store: 2 memory writes
- flow: 1 control flow operation

Your program must define a KernelBuilder class with a `build_kernel(forest_height, n_nodes,
batch_size, rounds)` method (and `debug_info()`) that builds an optimized program (list of VLIW
instructions in `self.instrs`) implementing the tree traversal algorithm.

The algorithm: for each element in a batch, traverse a binary tree for multiple rounds. At each
node, XOR the value with the node value, hash it, then go left or right based on parity. The
kernel must produce identical output to the reference.

Performance:
- Baseline (naive scalar): ~147,734 cycles
- Best known: ~1,363 cycles (108x speedup)

Key optimization opportunities:
1. VLIW: Pack multiple operations per cycle (up to 12 ALU, 6 VALU, 2 load, 2 store)
2. SIMD: Use vector operations (VLEN=8) to process 8 elements at once
3. Loop unrolling: Reduce loop overhead
4. Memory access patterns: Use vload/vstore for contiguous access
5. Instruction scheduling: Avoid dependencies, maximize ILP

Tips:
- Eval timeout is 120s. The kernel itself should run in under a second.
- Correctness is mandatory — incorrect output scores 0. The evaluation runs forest_height=10,
  rounds=16, batch_size=256 on 8 fresh random trees and reports the cycle count.
- Score = 147734 / cycles (the speed-up over the naive scalar kernel); higher is better.
- The simulator is frozen — you cannot modify it, only optimize how you build instructions.
- Focus on VLIW packing: fill all engine slots each cycle.
- Use SIMD (valu) operations to process 8 batch elements at once.
- The hash function has 6 stages with 3 ops each — look for ILP within and across stages.
- Fully unroll loops when possible to eliminate flow control overhead.
- Minimize data dependencies between consecutive instructions.
"""


def polyomino_statement() -> str:
    import judge

    return judge.statement().strip()


def polyomino_prompt() -> str:
    return ("You are an expert competitive programmer specializing in algorithmic optimization.\n\n"
            + POLYOMINO_TASK + "\n# PROBLEM STATEMENT (statement.txt)\n\n" + polyomino_statement() + "\n")


def kernel_prompt() -> str:
    return ("You are an expert performance engineer specializing in VLIW / SIMD kernel optimization.\n\n"
            + KERNEL_TASK)


def _config(system_message: str, *, language: str, max_len: int) -> dict:
    return {
        "language": language,
        "diff_based_generation": True,
        "max_iterations": 1000,
        "checkpoint_interval": 10,
        "max_solution_length": max_len,
        "llm": {
            "api_base": "https://openrouter.ai/api/v1",
            "models": [{"name": "openrouter/openai/gpt-5", "weight": 1.0}],
            "timeout": 600,
        },
        "prompt": {"system_message": system_message},
        "evaluator": {"timeout": 600, "max_retries": 1, "cascade_evaluation": False},
        "monitor": {"enabled": False},
        "human_feedback_enabled": False,
    }


class _Literal(str):
    pass


def _literal(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def main() -> None:
    yaml.add_representer(_Literal, _literal)
    header = ("# Generated by benchmarks/coral_tasks/make_configs.py -- edit that file, not this one.\n"
              "# Usage: run_bench.sh baseline --benchmark-dir {d} (see scripts/server/run_pk_queue.sh)\n")
    for name, prompt, lang, max_len in [
        ("polyominoes", polyomino_prompt(), "cpp", 60000),
        ("kernel_builder", kernel_prompt(), "python", 100000),
    ]:
        cfg = _config(_Literal(prompt), language=lang, max_len=max_len)
        out = HERE / name / "config.yaml"
        out.write_text(header.format(d=os.path.relpath(out.parent, HERE.parents[1]))
                       + yaml.dump(cfg, sort_keys=False, allow_unicode=True, width=1000))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
