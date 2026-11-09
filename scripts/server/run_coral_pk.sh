#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_coral_pk.sh — CORAL on run_pk_queue.sh's two tasks, Polyominoes and
# Kernel Engineering, side by side in two tmux lanes, 3 hours each (no eval
# cap), with every LLM call's tokens and cost recorded through CORAL's LiteLLM
# gateway -- run_coral_cp.sh's setup, with CORAL_SUITE=pk.
#
#   ./scripts/server/run_coral_pk.sh check              # offline pre-flight + `coral validate` (no LLM calls)
#   ./scripts/server/run_coral_pk.sh smoke   [p1 p2]    # optional: 4-minute live run (default p1) into GROUP=coral_smoke
#   ./scripts/server/run_coral_pk.sh start   [p1 p2]    # launch the lanes (default: both), detached
#   ./scripts/server/run_coral_pk.sh status  [p1 p2]    # state, best, evals, cost, tokens, wall
#   ./scripts/server/run_coral_pk.sh compare [p1 p2]    # writes comparison.{md,csv,json}
#   ./scripts/server/run_coral_pk.sh stop    [p1 p2]    # SIGTERM -> CORAL shuts its agents down
#
# Lanes:
#   p1  tmux coral_poly  polyominoes     score 0..100, graded by benchmarks/coral_tasks/polyominoes/judge.py
#   p2  tmux coral_kern  kernel_builder  CORAL's grader (cycles); best = 147734 / cycles
# Outputs: outputs/server/coral_exp/{polyominoes,kernel_builder}/. Knobs and
# details: scripts/server/run_coral_cp.sh --help (e.g. CORAL_MAX_EVALS=100).
# ---------------------------------------------------------------------------
CORAL_SUITE=pk exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_coral_cp.sh" "$@"
