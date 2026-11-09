#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_coral_cp2.sh — the multi-agent version of run_coral_cp.sh: CORAL with 2
# OpenCode agents (gpt-5) sharing one run on each circle-packing task, side by
# side in two tmux lanes, each until 100 graded attempts (both agents together)
# or 3 hours, whichever comes first -- run_coral_cp.sh's setup, with
# CORAL_SUITE=cp2.
#
#   ./scripts/server/run_coral_cp2.sh check              # offline pre-flight + `coral validate` (no LLM calls)
#   ./scripts/server/run_coral_cp2.sh start   [m1 m2]    # launch the lanes (default: both), detached
#   ./scripts/server/run_coral_cp2.sh status  [m1 m2]    # state, best, evals, cost, tokens, wall
#   ./scripts/server/run_coral_cp2.sh compare [m1 m2]    # writes comparison.{md,csv,json}
#   ./scripts/server/run_coral_cp2.sh stop    [m1 m2]    # SIGTERM -> CORAL shuts its agents down
#
# Lanes:
#   m1  tmux coral_cp_2ag   circle_packing       26 circles in the unit square      (gateway :4105)
#   m2  tmux coral_cpr_2ag  circle_packing_rect  21 circles, rectangle perimeter 4  (gateway :4106)
# Outputs: outputs/server/coral_exp_2ag/{circle_packing,circle_packing_rect}/.
# Knobs and details: scripts/server/run_coral_cp.sh --help (e.g. CORAL_AGENTS=4).
# ---------------------------------------------------------------------------
CORAL_SUITE=cp2 exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_coral_cp.sh" "$@"
