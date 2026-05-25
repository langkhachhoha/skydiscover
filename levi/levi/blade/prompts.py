"""BLADE prompt templates.

Operator prompts (mutate, crossover, repair) used by the small/mutation
workers, plus thin wrappers around LEVI's :mod:`levi.equilibrium.prompts`
and :mod:`levi.artifacts.code` strings so the bootstrap and paradigm-shift
phases reuse LEVI's hard-won prompt engineering.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..artifacts.code import DIVERSITY_SEED_PROMPT
from ..equilibrium.prompts import VARIANT_GENERATION_PROMPT
from ..simple.parser import OUTPUT_FORMAT_INSTRUCTION

__all__ = [
    "MUTATE_PROMPT",
    "CROSSOVER_PROMPT",
    "build_mutate_prompt",
    "build_crossover_prompt",
    "build_paradigm_prompt",
    "build_repair_prompt",
    "build_diverse_seed_prompt",
    "build_init_variant_prompt",
    "build_paradigm_variant_prompt",
    "build_meta_advice_prompt",
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
Write an improved version of the parent.
Treat the inspirations as ideas to draw from — do NOT copy their code (you
do not have it). Keep what works in the parent, change what doesn't.

### Critical requirements
1  Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. The function signature must match exactly what is specified
4. Ensure there are no syntax errors (matching parentheses, quotes, indentation)

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

### Critical requirements
1  Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. The function signature must match exactly what is specified
4. Ensure there are no syntax errors (matching parentheses, quotes, indentation)

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


# ---------------------------------------------------------------------------
# Bootstrap-phase prompt builders — mirror LEVI's Diversifier.
# ---------------------------------------------------------------------------


INIT_VARIANT_PROMPT = """\
# Init Variant

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Inspirations (existing diverse seeds — code + score)
{inspirations_block}

## Your task
Write an improved version of the function.

### Critical requirements
1  Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. The function signature must match exactly what is specified
4. Ensure there are no syntax errors (matching parentheses, quotes, indentation)

{format_instruction}
"""


def _seed_block(seeds: Sequence[tuple[str, float]]) -> str:
    """Render existing diverse seeds (code + score) — used as inspirations for
    init-variant prompts, NOT for paradigm-shift representatives."""
    if not seeds:
        return "(no seeds available yet)"
    parts: list[str] = []
    for idx, (code, score) in enumerate(seeds, start=1):
        parts.append(
            f"### Seed {idx} (score={score:.4f})\n```python\n{code}\n```"
        )
    return "\n\n".join(parts)


def build_diverse_seed_prompt(
    *,
    problem_description: str,
    function_signature: str,
    existing_seeds: Sequence[tuple[str, float]],
) -> str:
    """Frontier-model prompt for sequential diverse-seed generation.

    Mirrors LEVI's :data:`DIVERSITY_SEED_PROMPT` so the heavy model is
    pushed toward algorithmic diversity — each call sees all previously
    accepted seeds and is asked to design something *fundamentally
    different*. Appends BLADE's description-required format instruction.
    """
    existing_seeds_text = "\n\n---\n\n".join(
        f"### Seed {i + 1} (Score: {score:.17g}):\n```python\n{code}\n```"
        for i, (code, score) in enumerate(existing_seeds)
    )
    rendered = DIVERSITY_SEED_PROMPT.format(
        problem_title="Algorithm Optimization",
        problem_description=problem_description,
        function_signature=function_signature,
        existing_seeds=existing_seeds_text,
    )
    return rendered + "\n\n" + OUTPUT_FORMAT_INSTRUCTION


def build_init_variant_prompt(
    *,
    problem_description: str,
    function_signature: str,
    inspirations: Sequence[tuple[str, float]],
) -> str:
    """Mutation-model prompt for init-phase variant fanout.

    Each variant sees a small random sample of existing seeds (code+score)
    to nudge it toward *exploring around* one of the diverse paradigms.
    Mirrors LEVI's :meth:`build_init_variant_prompt`, adapted to BLADE's
    description-required output format.
    """
    return INIT_VARIANT_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        inspirations_block=_seed_block(inspirations),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
    )


# ---------------------------------------------------------------------------
# Meta-advisor — periodic short "lessons learnt" summary the mutation prompts
# can quote back to the model. Direct port of LEVI's cron-based advisor.
# ---------------------------------------------------------------------------


META_ADVICE_PROMPT = """\
# Lessons-Learned Advisor

You are reviewing the last batch of attempts on this optimisation problem.
Output a SHORT (3-5 sentences) note that future mutation prompts will
include verbatim. Focus on what to AVOID and what to TRY next — concrete,
prescriptive, code-shaped.

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Current state
- Best score so far: {best_score}
- Evaluations completed: {n_evaluations}
- Accept rate (last window): {accept_rate}
- Stagnation level: {stagnation_level} (0=fresh, 1=plateaued)

## Recent failure modes (top error messages, tail-truncated)
{error_block}

## Previous advice (carried over so you can refine, not repeat)
{previous_advice_block}

## Your task
Write the new advice block. No preamble, no markdown headers — just the
prescriptive prose. Examples of the kind of advice that's useful:
"Avoid recursion that can exceed Python's default depth on inputs > 1k;
prefer an explicit stack." or "Several attempts crashed on empty input;
add a guard for `n == 0` returning the trivial solution."

Keep it under 100 words. Do NOT restate the problem.
"""


def _error_block(recent_errors: Sequence[str]) -> str:
    if not recent_errors:
        return "(none in this period)"
    parts: list[str] = []
    for i, e in enumerate(recent_errors[:5], start=1):
        s = (e or "").strip().replace("\n", " ")
        if len(s) > 200:
            s = s[:200] + "…"
        parts.append(f"{i}. {s}")
    return "\n".join(parts)


def build_meta_advice_prompt(
    *,
    problem_description: str,
    function_signature: str,
    best_score: float,
    n_evaluations: int,
    accept_rate: float,
    stagnation_level: float,
    recent_errors: Sequence[str],
    previous_advice: str | None,
) -> str:
    """Render the meta-advisor prompt for the small (mutation) model.

    The mutation model is asked to compress the last period of attempts
    into a few sentences of prescriptive advice. The string this produces
    is then injected verbatim into the next batch of mutate / crossover
    prompts (via :func:`build_mutate_prompt`'s ``meta_advice`` slot).
    """
    if best_score == float("-inf") or not math.isfinite(best_score):
        best_score_str = "n/a"
    else:
        best_score_str = f"{best_score:.4f}"
    previous = (previous_advice or "").strip()
    previous_block = previous if previous else "(none yet — this is the first cycle)"
    return META_ADVICE_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        best_score=best_score_str,
        n_evaluations=n_evaluations,
        accept_rate=f"{accept_rate:.2f}",
        stagnation_level=f"{stagnation_level:.2f}",
        error_block=_error_block(recent_errors),
        previous_advice_block=previous_block,
    )


def build_paradigm_variant_prompt(
    *,
    problem_description: str,
    function_signature: str,
    base_code: str,
    base_score: float,
) -> str:
    """Mutation-model prompt for paradigm-shift variant fanout.

    Wraps LEVI's :data:`VARIANT_GENERATION_PROMPT` so the small model
    explores nearby regions around a fresh paradigm-shift solution.
    Appends BLADE's description-required format instruction.
    """
    rendered = VARIANT_GENERATION_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        base_code=base_code,
        base_score=base_score,
    )
    return rendered + "\n\n" + OUTPUT_FORMAT_INSTRUCTION


# ---------------------------------------------------------------------------
# Paradigm-shift prompt (BLADE Lite — one prompt, stagnation-aware framing)
# ---------------------------------------------------------------------------
#
# The previous version had three separate stage prompts (early/mid/late)
# routed by a budget/stagnation heuristic. Live runs showed the late-stage
# template ("surgical fix on best anchor") starved exploration exactly when
# the search needed it most, and the budget routing made the prompt depend
# on a value the model cannot see.
#
# This module exposes a single ``build_paradigm_prompt`` that always asks
# the frontier to *either* synthesise a stronger hybrid from the anchors,
# *or* propose a fundamentally different paradigm if synthesis is no longer
# productive. The frontier itself picks. A short numeric context block
# (best score, evaluations, stagnation level) is included so the model can
# calibrate without us hard-coding stage routing.


_PARADIGM_PROMPT = """\
# Algorithmic Paradigm Shift Challenge

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Archive Snapshot
The archive has run {n_evaluations} evaluations and currently occupies
{n_cells} behavioural cells (one cell ≈ one algorithmic paradigm — cells
are recomputed periodically from program AST features + description
embeddings, so each anchor below is the strongest representative of a
distinct paradigm).

Stagnation level (0 = just improved, 1 = stuck): {stagnation:.2f}.

### Anchor representatives (full code + description + score)
{anchor_block}

### Additional inspirations (description + score only — different paradigms whose code is withheld)
{inspiration_block}
{strategy_log_block}
## Your Task

Read the anchors above and decide *which* of the two moves is the better
bet, given the stagnation level:

**(A) SYNTHESISE** — combine 2-3 concrete mechanisms drawn from
different anchors into one structurally coherent program that beats each
of them individually. Prefer this when several anchors are close in
score (the archive has good ingredients; the missing piece is the
combination).

**(B) PARADIGM SHIFT** — design a fundamentally different algorithmic
approach (a paradigm class that does NOT appear in any anchor or
inspiration). Internal data structures, control flow, and termination
condition must all reflect the new paradigm. Prefer this when
synthesis would only produce a marginal mix.

Make the choice explicit in your description (start with "MOVE: A" or
"MOVE: B" on the first line of `## Description`). Whatever you pick:

- Match the function signature exactly.
- Include all imports.
- Avoid any strategy whose Strategy-Log entry already has delta ≤ 0 —
  that approach has been tried and did not improve over the best.
- Do NOT just retune constants in an anchor — that is not a paradigm
  shift, it is mutation, and the mutation worker is already doing it.

{format_instruction}
"""


def _anchor_block(anchors: Sequence[tuple[str, str, float]]) -> str:
    """Render anchor representatives with full code."""
    if not anchors:
        return "(archive too small — no anchors yet)"
    parts: list[str] = []
    for i, (code, desc, score) in enumerate(anchors, start=1):
        d = (desc or "").strip().replace("\n", " ")
        if len(d) > 400:
            d = d[:400].rstrip() + "…"
        parts.append(
            f"#### Anchor {i} (score={score:.4f})\n"
            f"_Description_: {d or '(no description)'}\n"
            f"```python\n{code}\n```"
        )
    return "\n\n".join(parts)


def build_paradigm_prompt(
    *,
    problem_description: str,
    function_signature: str,
    n_evaluations: int,
    n_cells: int,
    anchors: Sequence[tuple[str, str, float]],
    inspirations: Sequence[tuple[str, float]] = (),
    recent_trials: Sequence[str] = (),
    stagnation: float = 0.0,
) -> str:
    """Build the BLADE Lite paradigm-shift prompt.

    One prompt, no stage routing. ``stagnation`` and ``n_cells`` are
    surfaced in the prompt header so the frontier model can calibrate
    its own choice between synthesis (A) and paradigm shift (B).
    """
    if recent_trials:
        strategy_log_block = (
            "\n## Strategy Log (recent paradigm attempts)\n"
            + "\n".join(f"- {line}" for line in recent_trials)
            + "\n"
        )
    else:
        strategy_log_block = ""

    return _PARADIGM_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        n_evaluations=n_evaluations,
        n_cells=n_cells,
        stagnation=stagnation,
        anchor_block=_anchor_block(anchors),
        inspiration_block=_inspiration_block(inspirations),
        strategy_log_block=strategy_log_block,
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
    )
