#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_coral_pk2.sh — the multi-agent version of run_coral_pk.sh: CORAL with 2
# OpenCode agents (gpt-5) sharing one run on Polyominoes and on Kernel
# Engineering, side by side in two tmux lanes, 3 hours each (no eval cap: the
# CORAL paper matches multi-agent runs on wall-clock time) -- run_coral_cp.sh's
# setup, with CORAL_SUITE=pk2.
#
#   ./scripts/server/run_coral_pk2.sh check              # offline pre-flight + `coral validate` (no LLM calls)
#   ./scripts/server/run_coral_pk2.sh start   [q1 q2]    # launch the lanes (default: both), detached
#   ./scripts/server/run_coral_pk2.sh status  [q1 q2]    # state, best, evals, cost, tokens, wall
#   ./scripts/server/run_coral_pk2.sh compare [q1 q2]    # writes comparison.{md,csv,json}
#   ./scripts/server/run_coral_pk2.sh stop    [q1 q2]    # SIGTERM -> CORAL shuts its agents down
#
# Lanes:
#   q1  tmux coral_poly_2ag  polyominoes     score 0..100, graded by benchmarks/coral_tasks/polyominoes/judge.py (gateway :4107)
#   q2  tmux coral_kern_2ag  kernel_builder  CORAL's grader (cycles); best = 147734 / cycles             (gateway :4108)
# Outputs: outputs/server/coral_exp_2ag/{polyominoes,kernel_builder}/. Knobs and
# details: scripts/server/run_coral_cp.sh --help (e.g. CORAL_AGENTS=4).
# ---------------------------------------------------------------------------
CORAL_SUITE=pk2 exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_coral_cp.sh" "$@"
