"""Quick A/B test: does MUTATION_OUTPUT_FORMAT fix the parse_miss storm?

For N parallel calls to the Qwen mutation model on the exact circle_packing
init-variant prompt that failed in production, count how many produce a
parseable code block under:
  (A) old OUTPUT_FORMAT_INSTRUCTION (strict ## Description + ## Code)
  (B) new MUTATION_OUTPUT_FORMAT    (single fenced python block, example)

Usage:
    OPENAI_API_KEY=sk-or-... uv run --extra dev python scripts/test_mutation_format.py [N]
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
from pathlib import Path

# Make sure we use the in-tree levi package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "levi"))

from levi.blade.prompts import (  # type: ignore  # noqa: E402
    INIT_VARIANT_PROMPT,
    _seed_block,
)
from levi.simple.parser import OUTPUT_FORMAT_INSTRUCTION, OutputParser  # type: ignore  # noqa: E402

MODEL = "openrouter/qwen/qwen3-30b-a3b-instruct-2507"

# Force OpenRouter routing for litellm
os.environ.setdefault("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
if "OPENROUTER_API_KEY" not in os.environ and "OPENAI_API_KEY" in os.environ:
    os.environ["OPENROUTER_API_KEY"] = os.environ["OPENAI_API_KEY"]


# ---- Reproduce a realistic init-variant prompt for circle_packing ----------

sys.path.insert(0, str(ROOT / "levi" / "examples" / "circle_packing"))
from problem import (  # type: ignore  # noqa: E402
    PROBLEM_DESCRIPTION,
    FUNCTION_SIGNATURE,
)

# A minimal realistic seed program — exact contents don't matter, we just
# need a believable Python solution to anchor the inspirations block (mirrors
# what Phase 2 actually sees from Phase 1 frontier-model output).
_SEED = """\
import numpy as np

def place_circles_and_radii():
    n = 26
    # 5x5 grid + 1 extra in the centre row, equal radii by spacing
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append((0.1 + 0.2 * j, 0.1 + 0.2 * i))
    pts.append((0.5, 0.5))
    centers = np.array(pts, dtype=float)
    r = 0.09 * np.ones(n, dtype=float)
    return centers, r
"""

INSPIRATION_SEEDS = [
    (_SEED, 2.4389),
    (_SEED, 2.5730),
]


def build_prompt(format_instruction: str) -> str:
    return INIT_VARIANT_PROMPT.format(
        problem_description=PROBLEM_DESCRIPTION,
        function_signature=FUNCTION_SIGNATURE,
        inspirations_block=_seed_block(INSPIRATION_SEEDS),
        format_instruction=format_instruction,
    )


# ---- Litellm async call -----------------------------------------------------

import litellm  # type: ignore  # noqa: E402

litellm.drop_params = True


async def call_one(prompt: str, *, temperature: float = 0.9, max_tokens: int | None = 1200) -> str:
    kwargs: dict = dict(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    try:
        resp = await litellm.acompletion(**kwargs)
        return resp["choices"][0]["message"]["content"] or ""
    except Exception as e:
        return f"<<ERROR:{type(e).__name__}: {e}>>"


async def run_batch(label: str, prompt: str, n: int, max_tokens: int | None = 1200) -> dict:
    parser = OutputParser()
    mt_label = "None (provider default)" if max_tokens is None else str(max_tokens)
    print(f"\n=== {label} — N={n}  max_tokens={mt_label} ===")
    results = await asyncio.gather(*(call_one(prompt, max_tokens=max_tokens) for _ in range(n)))

    n_parsed_code = 0
    n_parsed_desc = 0
    n_error = 0
    samples_miss: list[str] = []

    for i, raw in enumerate(results, 1):
        if raw.startswith("<<ERROR:"):
            n_error += 1
            continue
        parsed = parser.parse(raw)
        if parsed.has_code:
            n_parsed_code += 1
        else:
            if len(samples_miss) < 2:
                samples_miss.append(raw[:300].replace("\n", "\\n"))
        if parsed.has_description:
            n_parsed_desc += 1

    pct_code = 100.0 * n_parsed_code / n if n else 0.0
    pct_desc = 100.0 * n_parsed_desc / n if n else 0.0
    print(f"  has_code:        {n_parsed_code}/{n}  ({pct_code:.1f}%)")
    print(f"  has_description: {n_parsed_desc}/{n}  ({pct_desc:.1f}%)")
    print(f"  errors:          {n_error}/{n}")
    if samples_miss:
        print("  parse_miss sample heads:")
        for s in samples_miss:
            print(f"    - {s[:200]}…")

    return {
        "label": label,
        "n": n,
        "has_code": n_parsed_code,
        "has_description": n_parsed_desc,
        "errors": n_error,
    }


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    random.seed(0)

    prompt_old = build_prompt(OUTPUT_FORMAT_INSTRUCTION)

    print(f"Model: {MODEL}")
    print(f"Prompt size: {len(prompt_old)} chars  |  using OLD strict 2-section format")

    res_1200 = await run_batch("OLD @ max_tokens=1200 (old default)", prompt_old, n, max_tokens=1200)
    res_4096 = await run_batch("OLD @ max_tokens=4096", prompt_old, n, max_tokens=4096)
    res_none = await run_batch("OLD @ max_tokens=None (new default)", prompt_old, n, max_tokens=None)

    print("\n=== Summary (has_code/N, has_description/N) ===")
    print(f"  max_tokens=1200            code={res_1200['has_code']}/{n}   desc={res_1200['has_description']}/{n}")
    print(f"  max_tokens=4096            code={res_4096['has_code']}/{n}   desc={res_4096['has_description']}/{n}")
    print(f"  max_tokens=None (default)  code={res_none['has_code']}/{n}   desc={res_none['has_description']}/{n}")


if __name__ == "__main__":
    asyncio.run(main())
