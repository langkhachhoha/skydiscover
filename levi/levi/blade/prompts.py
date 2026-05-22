"""BLADE prompt templates.

Two operator prompts (mutate, crossover) used by the mutation workers, plus
a thin wrapper around LEVI's :mod:`levi.equilibrium.prompts` so the frontier
(paradigm-shift) phase reuses LEVI's three-phase prompts unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..equilibrium.prompts import PARADIGM_SHIFT_PROMPTS, get_budget_stage
from ..simple.parser import OUTPUT_FORMAT_INSTRUCTION

__all__ = [
    "MUTATE_PROMPT",
    "CROSSOVER_PROMPT",
    "build_mutate_prompt",
    "build_crossover_prompt",
    "build_paradigm_prompt",
    "build_repair_prompt",
    "PARADIGM_SHIFT_PROMPTS",
    "get_budget_stage",
]


MUTATE_PROMPT = """\
# Mutate

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Parent solution
Score: {parent_score:.4f}
```python
{parent_code}
```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{meta_advice_block}\
## Your task
Produce a **mutated variant** of the parent that is meaningfully different.
Treat the inspirations as ideas to draw from — do NOT copy their code (you
do not have it). Keep what works in the parent, change what doesn't.

{format_instruction}
"""


CROSSOVER_PROMPT = """\
# Crossover

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Parent A
Score: {parent_a_score:.4f}
```python
{parent_a_code}
```

## Parent B
Score: {parent_b_score:.4f}
```python
{parent_b_code}
```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{meta_advice_block}\
## Your task
Produce a **hybrid solution** that combines the strongest mechanisms of
both parents while fixing at least one weakness. Be structural, not
stitched: do not paste A's branch into B's branch.

{format_instruction}
"""


REPAIR_PROMPT = """\
# Repair

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Broken candidate (parent score was {parent_score})
```python
{broken_code}
```

## Error
```
{error_msg}
```

## Your task
Produce a **corrected version** of the candidate that addresses the error
above. Keep the algorithmic intent intact — only patch what's broken.

{format_instruction}
"""


def _inspiration_block(inspirations: Sequence[tuple[str, float]]) -> str:
    """Render inspirations as ``description + score`` only (no code).

    Each element is ``(description, score)``. Returns an empty block when
    the sequence is empty so the prompt stays clean.
    """
    if not inspirations:
        return "(no inspirations available yet)"
    parts: list[str] = []
    for i, (desc, score) in enumerate(inspirations, start=1):
        d = (desc or "").strip().replace("\n", " ")
        if len(d) > 400:
            d = d[:400].rstrip() + "…"
        parts.append(f"{i}. (score={score:.3f}) {d}")
    return "\n".join(parts)


def _meta_advice_block(meta_advice: str | None) -> str:
    """Format the optional meta-advisor lesson block."""
    if not meta_advice:
        return ""
    return f"## Lessons learnt so far\n{meta_advice.strip()}\n\n"


def build_mutate_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_code: str,
    parent_score: float,
    inspirations: Sequence[tuple[str, float]],
    meta_advice: str | None = None,
) -> str:
    return MUTATE_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_code=parent_code,
        parent_score=parent_score,
        inspirations_block=_inspiration_block(inspirations),
        meta_advice_block=_meta_advice_block(meta_advice),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
    )


def build_crossover_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_a_code: str,
    parent_a_score: float,
    parent_b_code: str,
    parent_b_score: float,
    inspirations: Sequence[tuple[str, float]],
    meta_advice: str | None = None,
) -> str:
    return CROSSOVER_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_a_code=parent_a_code,
        parent_a_score=parent_a_score,
        parent_b_code=parent_b_code,
        parent_b_score=parent_b_score,
        inspirations_block=_inspiration_block(inspirations),
        meta_advice_block=_meta_advice_block(meta_advice),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
    )


def build_repair_prompt(
    *,
    problem_description: str,
    function_signature: str,
    broken_code: str,
    parent_score: float | None,
    error_msg: str,
) -> str:
    if parent_score is None or not math.isfinite(parent_score):
        parent_score_str = "n/a"
    else:
        parent_score_str = f"{parent_score:.4f}"
    return REPAIR_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        broken_code=broken_code,
        parent_score=parent_score_str,
        error_msg=error_msg[-1500:],  # tail is where the actual exception lives
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
    )


def build_paradigm_prompt(
    *,
    stage: str,
    problem_description: str,
    function_signature: str,
    n_evaluations: int,
    n_regions: int,
    representatives: Sequence[tuple[str, float]],
    recent_trials: Sequence[str],
) -> str:
    """Wrap LEVI's three-phase paradigm prompt.

    Representatives are description+score pairs (no code) — this is the
    key BLADE deviation from LEVI: the frontier sees ideas, not source.
    """
    template = PARADIGM_SHIFT_PROMPTS.get(stage, PARADIGM_SHIFT_PROMPTS["early"])
    rep_block = _inspiration_block(representatives)
    representative_solutions = (
        "### Behavioural sketches (description + score; code intentionally withheld)\n"
        f"{rep_block}\n"
    )
    if recent_trials:
        strategy_log_block = (
            "\n## Strategy Log (recent paradigm attempts)\n"
            + "\n".join(f"- {line}" for line in recent_trials)
            + "\n"
        )
    else:
        strategy_log_block = ""
    rendered = template.format(
        problem_description=problem_description,
        function_signature=function_signature,
        n_evaluations=n_evaluations,
        n_regions=n_regions,
        representative_solutions=representative_solutions,
        strategy_log_block=strategy_log_block,
    )
    return rendered + "\n\n" + OUTPUT_FORMAT_INSTRUCTION
