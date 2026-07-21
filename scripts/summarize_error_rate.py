#!/usr/bin/env python3
"""Render an ``error_rate_report.json`` as a GitHub-flavoured Markdown table.

Written for the ``Error-rate summary`` step of
``.github/workflows/ablation.yml``, which pipes stdout into
``$GITHUB_STEP_SUMMARY``. Runs on the stdlib only, so it works with the
runner's system ``python3`` without touching the LEVI environment.

Usage::

    python3 scripts/summarize_error_rate.py <report.json> [--mode with_advisor]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "n/a"


def render(report: dict, *, mode: str | None) -> str:
    windows = report.get("windows", [])
    title = f"SpecEvo error rate — mode `{mode}`" if mode else "SpecEvo error rate"
    advisor = "ON" if report.get("enable_meta_advice") else "OFF"

    out: list[str] = [
        f"## {title}",
        "",
        f"- Advisor: **{advisor}** (every {report.get('meta_advice_interval')} iterations)",
        f"- Single general mutation prompt: **{report.get('single_prompt_operators')}**",
        f"- Window: {report.get('window_evals')} iterations, origin at eval "
        f"{report.get('post_init_eval_count')} (end of init phase)",
        f"- Windows completed: **{len(windows)}**",
        f"- Overall error rate: **{_pct(report.get('overall_error_rate'))}** "
        f"({report.get('total_errors')}/{report.get('total_scored')})",
        f"- Excluded: {report.get('total_excluded_llm_failures')} failed LLM call(s) "
        "— no candidate was produced, so they are left out of the ratio",
        "",
        "| window | evals | scored | errors | error rate | skipped | best score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for w in windows:
        best = w.get("best_score")
        best_str = f"{best:.6f}" if isinstance(best, (int, float)) else "n/a"
        span = f"{w['eval_start']}→{w['eval_end']}"
        if w.get("final_partial"):
            span += " *(partial)*"
        out.append(
            f"| {w['window']} | {span} | {w['n_scored']} | {w['n_errors']} | "
            f"{_pct(w['error_rate'])} | {w['n_excluded']} | {best_str} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path, help="Path to error_rate_report.json")
    ap.add_argument("--mode", default=None, help="Ablation arm label for the heading")
    args = ap.parse_args()

    if not args.report.is_file():
        print(f"No error-rate report at `{args.report}` — "
              "the run did not reach its first measurement window.")
        return 0

    print(render(json.loads(args.report.read_text()), mode=args.mode))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
