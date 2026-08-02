#!/usr/bin/env python3
"""Print mean ± std cost only (math / system × GPT-5 / KIMI K2).

Sources:
  math_gpt.json, math_kimi.json, system_gpt.json, system_kimi.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent

DATASETS = [
    ("math_gpt.json", "Math — GPT-5"),
    ("math_kimi.json", "Math — KIMI K2"),
    ("system_gpt.json", "System — GPT-5"),
    ("system_kimi.json", "System — KIMI K2"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def per_task_cost_stats(data: dict) -> dict:
    """mean ± std (sample, ddof=1) of cost per task × method across seeds."""
    tasks = data["tasks"]
    methods = data["methods"]
    stats = {}
    for task in tasks:
        stats[task] = {}
        for method in methods:
            values = []
            for seed in data["seeds"]:
                for result in seed["results"]:
                    if result["task"] == task:
                        values.append(result[method]["cost"])
                        break
            arr = np.array(values, dtype=float)
            stats[task][method] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            }
    return stats


def total_cost_stats(data: dict) -> dict:
    """mean ± std of per-seed total cost (sum over tasks), per method."""
    methods = data["methods"]
    totals = {m: [] for m in methods}
    for seed in data["seeds"]:
        seed_total = {m: 0.0 for m in methods}
        for res in seed["results"]:
            for m in methods:
                seed_total[m] += res[m]["cost"]
        for m in methods:
            totals[m].append(seed_total[m])
    return {
        m: {
            "mean": float(np.mean(totals[m])),
            "std": float(np.std(totals[m], ddof=1)) if len(totals[m]) > 1 else 0.0,
        }
        for m in methods
    }


def print_dataset(label: str, data: dict) -> None:
    methods = data["methods"]
    task_stats = per_task_cost_stats(data)
    totals = total_cost_stats(data)

    print("=" * 90)
    print(label)
    print("=" * 90)

    header = f"{'Task':<32}" + "".join(f"{m:>14}" for m in methods)
    print(header)
    print("-" * 90)

    for task in data["tasks"]:
        row = f"{task:<32}"
        for m in methods:
            mean = task_stats[task][m]["mean"]
            std = task_stats[task][m]["std"]
            row += f"{mean:>6.2f}±{std:<5.2f}"
        print(row)

    print("-" * 90)
    total_row = f"{'TOTAL (sum over tasks)':<32}"
    for m in methods:
        mean = totals[m]["mean"]
        std = totals[m]["std"]
        total_row += f"{mean:>6.2f}±{std:<5.2f}"
    print(total_row)
    print()


def main() -> None:
    for filename, label in DATASETS:
        path = ROOT / filename
        if not path.exists():
            print(f"[skip] missing {filename}")
            continue
        print_dataset(label, load(path))


if __name__ == "__main__":
    main()
