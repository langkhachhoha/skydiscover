#!/usr/bin/env python3
"""In 5 mutation prompt từ mutation_prompts.json và prompt mặc định khi không bật prompt bank.

Chạy từ thư mục gốc project levi (có pyproject.toml)::

    uv run python examples/circle_packing/print_mutation_prompt_comparison.py

Hoặc::

    cd examples/circle_packing && uv run python print_mutation_prompt_comparison.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Package `levi` + module `problem` cục bộ của example
_EXAMPLE_DIR = Path(__file__).resolve().parent
_LEVI_ROOT = _EXAMPLE_DIR.parents[1]  # .../levi (pyproject.toml + package levi/)
if str(_LEVI_ROOT) not in sys.path:
    sys.path.insert(0, str(_LEVI_ROOT))
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

import problem  # noqa: E402
from levi.artifacts.code import CodeAdapter  # noqa: E402
from levi.config import BudgetConfig, LeviConfig, PipelineConfig  # noqa: E402
from levi.core import Program  # noqa: E402
from levi.prompts import ProgramWithScore  # noqa: E402


def _minimal_config(*, output_mode: str = "full") -> LeviConfig:
    def _dummy_score_fn(_code: str) -> dict:
        return {"score": 0.0}

    return LeviConfig(
        problem_description=problem.PROBLEM_DESCRIPTION.strip(),
        function_signature=problem.FUNCTION_SIGNATURE.strip(),
        seed_program="# seed",
        score_fn=_dummy_score_fn,
        budget=BudgetConfig(evaluations=1),
        pipeline=PipelineConfig(output_mode=output_mode),
    )


def _stub_parents() -> list[ProgramWithScore]:
    # Hai parent giống pipeline thật: v1 + một inspiration (score từ elite thường là số)
    p1 = Program(
        content=(
            "# v1 (parent)\n"
            "import numpy as np\n\n"
            "def run_packing() -> tuple[np.ndarray, np.ndarray, float]:\n"
            "    ...  # stub\n"
            "    return np.zeros((26, 2)), np.zeros(26), 0.0\n"
        )
    )
    p2 = Program(
        content=(
            "# v2 (inspiration)\n"
            "import numpy as np\n\n"
            "def run_packing() -> tuple[np.ndarray, np.ndarray, float]:\n"
            "    ...  # stub khác\n"
            "    return np.ones((26, 2)) * 0.5, np.ones(26) * 0.01, 0.26\n"
        )
    )
    return [ProgramWithScore(p1, None), ProgramWithScore(p2, None)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeat-nonbank",
        action="store_true",
        help="Sau mỗi template bank, in lại nguyên prompt build_mutation_prompt (mặc định chỉ in 1 lần cuối).",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Dùng pipeline.output_mode='diff' cho prompt mặc định (không bank).",
    )
    args = parser.parse_args()

    json_path = _EXAMPLE_DIR / "mutation_prompts.json"
    entries = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise SystemExit("mutation_prompts.json must be a JSON array")

    cfg = _minimal_config(output_mode="diff" if args.diff else "full")
    adapter = CodeAdapter(cfg)
    parents = _stub_parents()

    # Giống producer khi không bundle / không bank: không truyền SAL kwargs → không có Search Trajectory
    use_diff = cfg.pipeline.output_mode == "diff"
    default_prompt = adapter.build_mutation_prompt(
        parents,
        meta_advice=None,
        model=None,
        use_diff=use_diff,
    )

    print("=" * 88)
    print("PROMPT BANK (mutation_prompts.json) — mỗi id một template, đã fill placeholder thật")
    print("(meta_advice / feedback / SAL trajectory tắt để khớp default bên dưới)")
    print("=" * 88)

    for item in entries:
        pid = item.get("id", "?")
        text = item.get("text", "")
        print()
        print("#" * 88)
        print(f"### BANK: {pid}")
        print("#" * 88)
        filled = adapter.build_mutation_prompt_from_template(
            parents,
            text,
            meta_advice=None,
            feedback=None,
            best_score=None,
            evals_since_best=None,
            stagnation=None,
            top_failures=None,
        )
        print(filled)
        if args.repeat_nonbank:
            print()
            print("-" * 88)
            print(
                f">>> KHÔNG BANK (cùng nội dung cho mọi id): build_mutation_prompt, "
                f"output_mode={cfg.pipeline.output_mode!r}"
            )
            print("-" * 88)
            print(default_prompt)
        else:
            print()
            print(
                f">>> Không bank cho id này: dùng chung build_mutation_prompt (output_mode={cfg.pipeline.output_mode!r}) — xem khối cuối."
            )

    print()
    print("=" * 88)
    print("KHÔNG DÙNG PROMPT BANK — CodeAdapter.build_mutation_prompt() (PromptBuilder)")
    print("Một khung chung cho mọi arm; không có '# Mutation Task' / 'Your Task' riêng từng id.")
    print(f"pipeline.output_mode = {cfg.pipeline.output_mode!r}")
    print("=" * 88)
    if args.repeat_nonbank:
        print()
        print("(Đã in prompt mặc định kèm từng mục bank nhờ --repeat-nonbank — bỏ lặp ở đây.)")
    else:
        print()
        print(default_prompt)

    print()
    print("=" * 88)
    print("Ghi chú: thêm cờ --diff để xem ## Output dạng SEARCH/REPLACE; --repeat-nonbank in cặp đầy đủ sau mỗi id.")
    print("=" * 88)


if __name__ == "__main__":
    main()
