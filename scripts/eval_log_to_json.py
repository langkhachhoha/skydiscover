#!/usr/bin/env python3
"""Fold ``eval_code_log.jsonl`` into the single ``eval_code_log.json``.

Both runners write the JSON view themselves when a run ends normally. This is
for the runs that never got there — killed with ``tmux kill-session``, cut off
by a wall-clock ``--timeout``, or still going. The JSONL is flushed line by
line, so it is complete up to the moment the run stopped.

Usage::

    python scripts/eval_log_to_json.py outputs/server/relay_..._seed1_2026...
    python scripts/eval_log_to_json.py outputs/server/*/eval_code_log.jsonl
    python scripts/eval_log_to_json.py outputs/server --all     # recurse
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NAME = "eval_code_log.jsonl"


def _resolve(target: Path, recurse: bool) -> list[Path]:
    if target.is_file():
        return [target]
    if not target.is_dir():
        return []
    direct = target / NAME
    if direct.is_file() and not recurse:
        return [direct]
    return sorted(target.rglob(NAME))


def convert(jsonl_path: Path) -> tuple[Path, int]:
    records = []
    malformed = 0
    for line in jsonl_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # A run killed mid-write can leave one truncated last line.
            malformed += 1
    by_status: dict[str, int] = {}
    for record in records:
        key = str(record.get("status"))
        by_status[key] = by_status.get(key, 0) + 1

    run_dir = jsonl_path.parent
    meta: dict = {}
    for sidecar in ("run_config.json", "summary.json", "relay_summary.json"):
        path = run_dir / sidecar
        if path.is_file():
            try:
                meta[sidecar] = json.loads(path.read_text())
            except json.JSONDecodeError:
                pass

    out = jsonl_path.with_suffix(".json")
    out.write_text(
        json.dumps(
            {
                "schema": "eval-code-log/1",
                "converted_from": str(jsonl_path),
                "incomplete_run": True,
                "malformed_lines": malformed,
                "meta": meta,
                "n_records": len(records),
                "by_status": by_status,
                "records": records,
            },
            indent=2,
            default=str,
        )
    )
    return out, len(records)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help=f"Run directories or {NAME} files.")
    ap.add_argument("--all", action="store_true", help="Recurse into directories.")
    args = ap.parse_args()

    paths: list[Path] = []
    for raw in args.targets:
        found = _resolve(Path(raw), args.all)
        if not found:
            print(f"warning: no {NAME} under {raw}", file=sys.stderr)
        paths.extend(found)

    if not paths:
        print("Nothing to convert.", file=sys.stderr)
        return 1
    for path in paths:
        out, n = convert(path)
        print(f"{n:>6} records  ->  {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
