"""BLADE prompt templates.

Operator prompts used by the mutation / paradigm workers, plus thin
wrappers around LEVI's :mod:`levi.equilibrium.prompts` and
:mod:`levi.artifacts.code` strings for the bootstrap phase.

Compared to the previous version, three things changed:

1. **Mutate / crossover prompts ask for explicit analysis first.** The
   model writes a short ``# Analysis`` section (strengths, weaknesses,
   chosen target) *before* the code, so it commits to a hypothesis
   rather than randomly rewriting whatever it sees.

2. **Multiple prompt templates per operator.** Three mutate templates
   (general improvement, focused weakness fix, mechanism swap) and two
   crossover templates (structural hybrid, targeted component swap)
   are exposed via :class:`PromptSampler`. Every call draws one
   template uniformly at random — no learned weights — so the
   mutation model sees prompt-level diversity even when the parent
   pool is narrow. The :data:`TARGETED_MUTATE_PROMPT` is a separate
   template only used when a cached LLM-generated analysis is
   available.

3. **Three paradigm-shift modes.** ``build_synthesis_prompt`` (2-3
   anchors, hybridise close contenders), ``build_paradigm_shift_prompt``
   (2 anchors, propose a genuinely new paradigm class), and
   ``build_surgical_exploit_prompt`` (1 anchor = best, top descriptions
   as inspiration, deep micro-improvement of the current champion).
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..artifacts.code import DIVERSITY_SEED_PROMPT
from ..equilibrium.prompts import VARIANT_GENERATION_PROMPT
from ..simple.parser import OUTPUT_FORMAT_INSTRUCTION

__all__ = [
    "MUTATE_PROMPTS",
    "CROSSOVER_PROMPTS",
    "TARGETED_MUTATE_PROMPT",
    "PromptSampler",
    "build_mutate_prompt",
    "build_crossover_prompt",
    "build_targeted_mutate_prompt",
    "build_analysis_prompt",
    "build_synthesis_prompt",
    "build_paradigm_shift_prompt",
    "build_surgical_exploit_prompt",
    "build_repair_prompt",
    "build_diverse_seed_prompt",
    "build_init_variant_prompt",
    "build_paradigm_variant_prompt",
    "build_meta_advice_prompt",
    "classify_error",
]


def classify_error(msg: str) -> str:
    """Public wrapper around :func:`_classify_error`. See module docs."""
    return _classify_error(msg)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _inspiration_block(inspirations: Sequence[tuple[str, float]]) -> str:
    """Render inspirations as ``description + score`` only (no code)."""
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


# Reminder injected immediately AFTER OUTPUT_FORMAT_INSTRUCTION in every
# analysis-augmented prompt. The base instruction says "output must
# contain `## Description` and `## Code` in order" — we extend that
# contract with an extra `## Analysis` section that comes FIRST and is
# ignored by the parser (so the embedded description stays a clean
# 2-4 sentence paragraph). Without this reminder the model tends to
# either fold the analysis bullets into `## Description` (which then
# corrupts the embedding used for cell assignment) or skip them
# entirely.
ANALYSIS_OUTPUT_ORDER_HINT = """\

## Output sections (final reminder)
Your output MUST contain THREE sections, in this exact order:

1. ``## Analysis`` — the structured reasoning requested above. This
   section is read by the model on subsequent passes but is NOT used
   to embed the program into the search archive.
2. ``## Description`` — the short paragraph defined by the output
   format spec above (2-4 sentences, ≤ 80 words, plain prose, no
   bullet points). This IS what the archive embeds, so do not paste
   the analysis bullets here.
3. ``## Code`` — the fenced ```python``` program block.

Do not merge sections. Do not skip ``## Description``.
"""


def _anchor_block(anchors: Sequence[tuple[str, str, float]]) -> str:
    """Render anchor representatives with full code + description."""
    if not anchors:
        return "(no anchors available yet)"
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


# ---------------------------------------------------------------------------
# Mutate prompt templates — three variants, all ask for explicit analysis.
# ---------------------------------------------------------------------------


MUTATE_PROMPT_GENERAL = """\
# Mutate — Improvement Pass

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

Write an **improved** version of the parent. Before producing the
final output, write a ``## Analysis`` section (it is for your own
reasoning — the search system ignores it for archiving but reads it
back to you on future passes). It must have these four short
sub-sections:

1. **Components.** List the 3-5 main components / phases of the
   parent program (e.g. "initialisation grid", "outer SA loop",
   "neighbour move", "feasibility repair", "shrink step"). One line
   each.
2. **Strengths.** Which of those components are clearly working —
   i.e. you would NOT change them?
3. **Weaknesses.** Which component is most likely the reason the
   score is not higher? Cite the specific variable, constant, or
   routine.
4. **Plan.** One sentence: what concrete change you will make in the
   new program (e.g. "replace the linear cooling schedule in
   `cool()` with a geometric one, leave the rest unchanged").

Then write the improved program. Keep what works in the parent,
change what doesn't. Treat the inspirations as ideas only — you do
not have their code, so do not copy.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors (matching parentheses, quotes, indentation)
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


MUTATE_PROMPT_FOCUSED_FIX = """\
# Mutate — Focused Weakness Fix

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

The parent is mostly working. Your job is to **pick exactly ONE
weakness** and fix it surgically — leave everything else untouched.

First, write a ``## Analysis`` section with exactly these three
sub-sections:

1. **Tightest constraint.** What is the single most limiting factor
   in the parent's score? Be specific: name the variable / loop /
   constant. Examples: "the SA cooling rate decays too fast at
   step≈5000, premature freeze", "the perimeter-projection step
   shrinks all circles uniformly, wasting slack", "the initial hex
   grid leaves a triangular gap at the top-right corner".
2. **Fix.** The minimum change that addresses (1). One sentence.
3. **Preserved.** A 1-line confirmation of which top-level routines
   remain byte-for-byte identical to the parent.

Then output the program implementing exactly that one change. Do NOT
also retune unrelated constants. Do NOT also add new helper functions
unless they are the fix itself.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


MUTATE_PROMPT_MECHANISM_SWAP = """\
# Mutate — Mechanism Swap

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

The parent has one or more **interchangeable mechanisms** — places
where a different algorithmic choice could fit the same slot. Identify
one such mechanism and **swap it** for a better alternative. The
overall algorithm class stays the same; one sub-mechanism changes.

Examples of valid mechanism swaps:
- replace random uniform initialisation with a quasi-random sequence
- swap fixed step size for line search
- replace greedy neighbour acceptance with Metropolis acceptance
- swap a `while` convergence test for a `for` budgeted loop

First, write a ``## Analysis`` section with exactly these three
sub-sections:

1. **Mechanism identified.** Which slot in the parent are you
   targeting? Quote the function / block name.
2. **Old vs new.** "Currently uses X; replacing with Y because Z."
3. **Risk.** One sentence on what could go wrong with the swap and
   how the new code guards against it.

Then write the program. The signature, the I/O contract, and the rest
of the parent's structure must remain stable — only the swapped
mechanism changes.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


# Targeted mutation — only fired when an LLM-generated analysis already
# exists for this parent (see :meth:`BladeOrchestrator._analyze_parent`).
TARGETED_MUTATE_PROMPT = """\
# Targeted Mutate (analysis-guided)

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

## Analysis of this parent (produced earlier by a review pass)
{analysis}

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{meta_advice_block}\
## Your task

Read the analysis above and pick **one** of the suggested changes —
the one you judge most likely to actually increase the score on this
problem. Implement exactly that change.

First, write a ``## Analysis`` section with exactly these three
sub-sections:

1. **Chosen bottleneck.** Quote the bottleneck from the analysis
   above that you are targeting.
2. **Implementation plan.** One sentence describing the concrete
   code change.
3. **What stays unchanged.** A short list of routines / constants
   you are keeping byte-for-byte identical to the parent.

Then write the improved program. Do NOT change everything — make
exactly ONE structural change. The rest of the parent is, by
assumption, already doing its job.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


# ---------------------------------------------------------------------------
# Crossover prompt templates — two variants.
# ---------------------------------------------------------------------------


CROSSOVER_PROMPT_STRUCTURAL = """\
# Crossover — Structural Hybrid

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

Produce a **structural hybrid** that takes the strongest mechanism
from each parent and re-integrates them into one coherent program.
Stitching (paste a block from A inside a loop from B) is NOT what we
want — every routine in your output must make sense in the context of
the others.

First, write a ``## Analysis`` section with exactly these three
sub-sections:

1. **Component table.** A 3-row mapping like:
   - Initialisation: from A — reason …
   - Optimisation core: from B — reason …
   - Constraint handling: hybrid — reason …
2. **Compatibility note.** One sentence on how you reconciled any
   interface mismatch between the components (e.g. "A uses (x, y, r)
   tuples; B uses an Nx3 array — I converted A's output to the array
   form before passing into B's loop").
3. **Expected improvement.** One sentence on why this combination
   should beat both parents.

Then write the program. It must be self-contained: every helper
function used inside must be defined here. Do NOT reference
function names from A or B without re-defining them.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


CROSSOVER_PROMPT_COMPONENT_SWAP = """\
# Crossover — Targeted Component Swap

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Base parent (we keep this skeleton)
Score: {parent_a_score:.4f}
```python
{parent_a_code}
```

## Donor parent (we steal ONE component from this)
Score: {parent_b_score:.4f}
```python
{parent_b_code}
```

## Inspirations (paradigm sketches from the archive — descriptions only)
{inspirations_block}

{meta_advice_block}\
## Your task

Treat the base parent as the **skeleton** — keep its overall control
flow. Identify ONE component in the donor parent that is clearly
better than the base parent's equivalent, and **transplant it**.
Adapt the surrounding glue minimally so the transplanted component
fits.

First, write a ``## Analysis`` section with exactly these three
sub-sections:

1. **Donor component chosen.** What did you take from the donor
   (name the routine / block / constant) and why is it better than
   the base parent's equivalent?
2. **Glue work.** What did you have to change in the base parent to
   make the donor component fit? (Variable names, data layout, call
   sites.)
3. **What is NOT changed.** Confirm that the base parent's
   initialisation, optimisation loop body, and termination condition
   are otherwise untouched (or explain the one place where they had
   to change).

Then write the resulting program.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. No syntax errors
4. The complete, self-contained program goes inside one ```python``` fence

{format_instruction}
{analysis_output_order_hint}\
"""


# ---------------------------------------------------------------------------
# Prompt sampler — uniform random over the variant templates above.
# ---------------------------------------------------------------------------


@dataclass
class PromptSampler:
    """Picks one mutate / crossover prompt variant uniformly at random.

    No learning, no Thompson sampling — the variants are roughly
    equally useful and the mutation model benefits from prompt-level
    diversity regardless of which template wins on a given parent.
    """

    mutate_templates: list[str] = field(
        default_factory=lambda: [
            MUTATE_PROMPT_GENERAL,
            MUTATE_PROMPT_FOCUSED_FIX,
            MUTATE_PROMPT_MECHANISM_SWAP,
        ]
    )
    crossover_templates: list[str] = field(
        default_factory=lambda: [
            CROSSOVER_PROMPT_STRUCTURAL,
            CROSSOVER_PROMPT_COMPONENT_SWAP,
        ]
    )

    def pick_mutate(self, rng: random.Random) -> tuple[str, str]:
        """Return (label, template) for one mutate variant."""
        labels = ["general", "focused_fix", "mechanism_swap"]
        idx = rng.randrange(len(self.mutate_templates))
        return labels[idx], self.mutate_templates[idx]

    def pick_crossover(self, rng: random.Random) -> tuple[str, str]:
        """Return (label, template) for one crossover variant."""
        labels = ["structural", "component_swap"]
        idx = rng.randrange(len(self.crossover_templates))
        return labels[idx], self.crossover_templates[idx]


# Public aliases so callers can introspect / extend the variant set.
MUTATE_PROMPTS = {
    "general": MUTATE_PROMPT_GENERAL,
    "focused_fix": MUTATE_PROMPT_FOCUSED_FIX,
    "mechanism_swap": MUTATE_PROMPT_MECHANISM_SWAP,
}
CROSSOVER_PROMPTS = {
    "structural": CROSSOVER_PROMPT_STRUCTURAL,
    "component_swap": CROSSOVER_PROMPT_COMPONENT_SWAP,
}


def build_mutate_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_code: str,
    parent_score: float,
    inspirations: Sequence[tuple[str, float]],
    meta_advice: str | None = None,
    template: str | None = None,
) -> str:
    tmpl = template if template is not None else MUTATE_PROMPT_GENERAL
    return tmpl.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_code=parent_code,
        parent_score=parent_score,
        inspirations_block=_inspiration_block(inspirations),
        meta_advice_block=_meta_advice_block(meta_advice),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
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
    template: str | None = None,
) -> str:
    tmpl = template if template is not None else CROSSOVER_PROMPT_STRUCTURAL
    return tmpl.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_a_code=parent_a_code,
        parent_a_score=parent_a_score,
        parent_b_code=parent_b_code,
        parent_b_score=parent_b_score,
        inspirations_block=_inspiration_block(inspirations),
        meta_advice_block=_meta_advice_block(meta_advice),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


def build_targeted_mutate_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_code: str,
    parent_score: float,
    analysis: str,
    inspirations: Sequence[tuple[str, float]],
    meta_advice: str | None = None,
) -> str:
    return TARGETED_MUTATE_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_code=parent_code,
        parent_score=parent_score,
        analysis=analysis.strip(),
        inspirations_block=_inspiration_block(inspirations),
        meta_advice_block=_meta_advice_block(meta_advice),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


# ---------------------------------------------------------------------------
# Analysis prompt — produced by the mutation model and cached per parent.
# ---------------------------------------------------------------------------


ANALYSIS_PROMPT = """\
# Code Review — Bottleneck Identification

## Problem
{problem_description}

## Function signature
```python
{function_signature}
```

## Program under review (score={parent_score:.4f})
_Description_: {parent_description}
```python
{parent_code}
```

## Your task

Review this program and write **three** short sections. Be concrete:
cite variable names, line ranges, and magic constants. Do NOT propose
code — only analysis text.

### Algorithm summary
One or two sentences naming the algorithmic class and the key data
structures.

### Top 3 bottlenecks (ranked by expected impact)
What are the 3 most plausible reasons this program does NOT score
higher? For each, name the specific component (function, loop,
constant) responsible.

### Suggested changes
Three concrete, actionable changes — one per bottleneck. Each must be
specific enough that a competent programmer could implement it in
under 10 minutes without further questions. The three changes should
differ in *kind* (don't list three constant-tweaks).

Keep the whole review under 250 words. No code blocks.
"""


def build_analysis_prompt(
    *,
    problem_description: str,
    function_signature: str,
    parent_code: str,
    parent_score: float,
    parent_description: str,
) -> str:
    desc = (parent_description or "").strip() or "(no description)"
    return ANALYSIS_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        parent_code=parent_code,
        parent_score=parent_score,
        parent_description=desc,
    )


# ---------------------------------------------------------------------------
# Repair prompt — unchanged.
# ---------------------------------------------------------------------------


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
        error_msg=error_msg[-1500:],
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
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. The function signature must match exactly what is specified
4. Ensure there are no syntax errors (matching parentheses, quotes, indentation)

{format_instruction}
"""


def _seed_block(seeds: Sequence[tuple[str, float]]) -> str:
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
    return INIT_VARIANT_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        inspirations_block=_seed_block(inspirations),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
    )


# ---------------------------------------------------------------------------
# Meta-advisor
#
# The advisor periodically writes a short prescriptive note that future
# mutation prompts include verbatim. The prompt is split into three
# sections by design (What's working / What to try next / What to avoid)
# because earlier versions that only fed in raw failure messages produced
# defensive-only output ("avoid X", "clamp Y") and never told the model
# which existing approach to amplify. By forcing a "what's working"
# bucket and feeding in (a) descriptions of the top archived programs,
# (b) recent admits with their parent-delta and source operator, and
# (c) a small typed error taxonomy, the advisor sees both success and
# failure signal and can issue actionable forward-looking guidance.
# ---------------------------------------------------------------------------


META_ADVICE_PROMPT = """\
# Lessons-Learned Advisor

You are reviewing the search trajectory on this optimisation problem and
writing a short prescriptive note that the mutation model will read
verbatim before each future attempt. Your job is NOT to restate the
problem or repeat constraints the grader already enforces — it is to
amplify what is working, point at the next concrete thing to try, and
call out the few anti-patterns that are actually costing evaluations.

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

## Top archived programs (descriptions only — these are the leaders)
{top_descriptions_block}

## Recent admits (which operators are paying off, with score delta vs parent)
{recent_admits_block}

## Recent failure taxonomy (errors grouped by type, this window)
{error_taxonomy_block}

## Previous advice (carried over so you can refine, not repeat)
{previous_advice_block}

## Your task
Write the new advice block using EXACTLY these three short sections, in
this order, with these literal headers and no other markdown:

WORKING: <1-2 sentences naming the concrete approach / structure / trick
that the leaders share. Cite the operator (mutate_focused_fix,
crossover_component_swap, …) when one is clearly dominating admits. If
nothing is clearly working yet, say so plainly.>

TRY NEXT: <2-3 short imperative suggestions, ordered by priority. Be
code-shaped (mention specific data structures, algorithms, numerical
ranges, library calls). Build on WORKING — do NOT propose a tangent.>

AVOID: <1-2 anti-patterns that have actually shown up in this window's
failures, referencing the taxonomy. Skip generic defensive advice
(constraint checks that the grader already enforces, type validations,
etc.) unless they appear repeatedly here.>

Total length: under 120 words. No preamble. No extra headers. No bullet
characters — write each section as a single short paragraph after its
header.
"""


_ERROR_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("timeout", ("timeout", "exceeded", "timed out")),
    ("syntax", ("syntaxerror", "invalid syntax", "invalid character", "unexpected eof")),
    ("constraint", ("overlap", "not contained", "negative radius", "infeasible", "violat")),
    ("shape_mismatch", ("broadcast", "shape", "dimension", "size mismatch", "too many indices", "at least 2-d")),
    ("numpy_api", ("minimum() takes", "minimum() got", "unexpected keyword", "positional argument", "ufunc")),
    ("name_or_attr", ("not defined", "has no attribute", "is not associated", "nonetype")),
    ("type_error", ("unsupported operand", "must be", "expected", "argument", "cannot")),
)


def _classify_error(msg: str) -> str:
    low = msg.lower()
    for label, keys in _ERROR_PATTERNS:
        for k in keys:
            if k in low:
                return label
    return "other"


def _error_taxonomy_block(errors_by_source: Sequence[tuple[str, str]]) -> str:
    """Group errors into (kind, source) buckets with counts + one example.

    ``errors_by_source`` is a sequence of ``(source, error_message)`` tuples
    from the current window. We aggregate counts per error kind and remember
    the most-recent message of each kind as the human-readable example.
    """
    if not errors_by_source:
        return "(no failures in this window)"
    buckets: dict[str, dict[str, object]] = {}
    for src, msg in errors_by_source:
        kind = _classify_error(msg or "")
        b = buckets.setdefault(kind, {"count": 0, "sources": {}, "example": ""})
        b["count"] = int(b["count"]) + 1
        srcs = b["sources"]
        assert isinstance(srcs, dict)
        srcs[src] = int(srcs.get(src, 0)) + 1
        clean = (msg or "").strip().replace("\n", " ")
        if len(clean) > 160:
            clean = clean[:160].rstrip() + "…"
        b["example"] = clean
    rows = sorted(buckets.items(), key=lambda kv: -int(kv[1]["count"]))
    lines: list[str] = []
    for kind, b in rows:
        srcs = b["sources"]
        assert isinstance(srcs, dict)
        src_str = ", ".join(f"{s}×{c}" for s, c in sorted(srcs.items(), key=lambda kv: -kv[1]))
        lines.append(f"- {kind} ×{b['count']}  [{src_str}]  e.g.: {b['example']}")
    return "\n".join(lines)


def _top_descriptions_block(top_descriptions: Sequence[tuple[str, float]]) -> str:
    """Render top-K archived programs by score, descriptions only."""
    if not top_descriptions:
        return "(archive empty)"
    parts: list[str] = []
    for i, (desc, score) in enumerate(top_descriptions, start=1):
        d = (desc or "").strip().replace("\n", " ")
        if len(d) > 280:
            d = d[:280].rstrip() + "…"
        parts.append(f"{i}. (score={score:.4f}) {d}")
    return "\n".join(parts)


def _recent_admits_block(
    recent_admits: Sequence[tuple[str, float, float | None]],
) -> str:
    """Render recent admits as ``source | score | Δ vs parent``."""
    if not recent_admits:
        return "(no admits in this window)"
    parts: list[str] = []
    for source, score, delta in recent_admits:
        if delta is None or not math.isfinite(delta):
            delta_str = "Δ=n/a"
        else:
            sign = "+" if delta >= 0 else ""
            delta_str = f"Δ={sign}{delta:.4f}"
        parts.append(f"- {source:<26}  score={score:.4f}  {delta_str}")
    return "\n".join(parts)


def build_meta_advice_prompt(
    *,
    problem_description: str,
    function_signature: str,
    best_score: float,
    n_evaluations: int,
    accept_rate: float,
    stagnation_level: float,
    top_descriptions: Sequence[tuple[str, float]] = (),
    recent_admits: Sequence[tuple[str, float, float | None]] = (),
    errors_by_source: Sequence[tuple[str, str]] = (),
    previous_advice: str | None = None,
) -> str:
    """Build the advisor prompt.

    ``top_descriptions``  -- ``[(description, score), ...]`` for the top-K
                              archived programs (typically K=3).
    ``recent_admits``     -- ``[(source, score, delta_vs_parent or None),
                              ...]`` for the last few admits in this window.
    ``errors_by_source``  -- ``[(source, error_message), ...]`` from this
                              window; will be classified and aggregated.
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
        top_descriptions_block=_top_descriptions_block(top_descriptions),
        recent_admits_block=_recent_admits_block(recent_admits),
        error_taxonomy_block=_error_taxonomy_block(errors_by_source),
        previous_advice_block=previous_block,
    )


def build_paradigm_variant_prompt(
    *,
    problem_description: str,
    function_signature: str,
    base_code: str,
    base_score: float,
) -> str:
    rendered = VARIANT_GENERATION_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        base_code=base_code,
        base_score=base_score,
    )
    return rendered + "\n\n" + OUTPUT_FORMAT_INSTRUCTION


# ---------------------------------------------------------------------------
# Paradigm-shift prompts — three modes, dispatched by the orchestrator.
# ---------------------------------------------------------------------------
#
# Each mode receives a different ANCHOR configuration and asks the
# frontier model for a different kind of move:
#
#   • synthesis  — 2-3 anchors, hybridise close contenders.
#   • shift      — 2 anchors, design a fundamentally new paradigm.
#   • surgical   — 1 anchor = best, top descriptions as inspiration;
#                  deep tuning of the current champion.
#
# The orchestrator picks the mode based on stagnation level (low →
# synthesis, mid → shift, high → surgical) and supplies the right
# number of anchors.


SYNTHESIS_PROMPT = """\
# Paradigm Synthesis Challenge

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Archive Snapshot
The archive has run {n_evaluations} evaluations and currently occupies
{n_cells} behavioural cells. Stagnation level is {stagnation:.2f}
(0 = just improved, 1 = stuck). The search is **mildly stalled**:
several anchors are close in score but no single mutation has
combined their strengths.

### Top anchors (close-in-score, full code)
{anchor_block}

### Other paradigm inspirations (description + score only)
{inspiration_block}
{strategy_log_block}
## Your Task

Your job is **synthesis**, not invention. Read the anchors and write
ONE new program that combines 2-3 concrete mechanisms drawn from
different anchors into a structurally coherent whole, beating each of
them individually.

First, write a ``## Analysis`` section. Its first line MUST be
``MOVE: SYNTHESIS``. After that line, include exactly these three
sub-sections:

1. **Component table.** A mapping like:
   - Initialisation: from Anchor X (reason …)
   - Optimisation core: from Anchor Y (reason …)
   - Constraint repair: hybrid (reason …)
2. **Coherence note.** A sentence describing how the borrowed
   components share data — variable layout, units, call ordering.
   This is the hard part of synthesis: avoid Frankenstein code.
3. **Why this should beat all anchors.** One sentence per anchor:
   "beats anchor X because …".

Then write the program. Avoid any strategy whose Strategy-Log entry
has delta ≤ 0 — that approach has already failed. Do NOT just retune
constants in one anchor (the mutation worker is already doing that).

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. Every helper function must be defined here (don't reference functions
   from the anchors by name unless you copy their definition)

{format_instruction}
{analysis_output_order_hint}\
"""


PARADIGM_SHIFT_PROMPT = """\
# Paradigm Shift Challenge — Genuinely New Approach

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Archive Snapshot
The archive has run {n_evaluations} evaluations and currently occupies
{n_cells} behavioural cells. Stagnation level is {stagnation:.2f} —
the search is **moderately stalled**, suggesting the current paradigm
family has been mined out.

### Strongest paradigms currently in the archive (full code)
{anchor_block}

### Other paradigm inspirations (description + score only)
{inspiration_block}
{strategy_log_block}
## Your Task

Design a **fundamentally different algorithmic approach** — a
paradigm class that does NOT appear in any anchor or inspiration
above. The new program's internal data structures, control flow, and
termination condition must all reflect the new paradigm.

Concrete forbidden moves:
- Re-running the same algorithm with new constants.
- Stitching a sub-routine from one anchor onto another (that is
  synthesis, not a shift).
- Renaming variables in an existing anchor.

First, write a ``## Analysis`` section. Its first line MUST be
``MOVE: SHIFT``. After that line, include exactly these four
sub-sections:

1. **Paradigm name.** The textbook name of the algorithm class you
   are proposing (e.g. "Lloyd relaxation", "Power diagram packing",
   "Lagrangian relaxation with subgradient ascent", "Branch-and-cut
   over a conflict graph").
2. **Why this paradigm fits the problem.** Two sentences. Cite the
   specific problem feature that the paradigm exploits.
3. **Why current anchors miss it.** One sentence: what assumption
   the anchors share that the new paradigm drops.
4. **Risk.** One sentence on the most likely implementation pitfall
   and how your code avoids it.

Then write the complete, runnable program. Avoid any strategy whose
Strategy-Log entry has delta ≤ 0 — that approach has been tried.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. Implement the paradigm yourself — do not assume any non-stdlib
   exotic library is available beyond numpy / scipy.

{format_instruction}
{analysis_output_order_hint}\
"""


SURGICAL_EXPLOIT_PROMPT = """\
# Surgical Exploit Challenge — Tune the Champion

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Archive Snapshot
The archive has run {n_evaluations} evaluations and currently occupies
{n_cells} behavioural cells. Stagnation level is {stagnation:.2f} —
the search is **deeply stalled**. The same family of solutions has
dominated for many evaluations, and previous paradigm attempts have
not produced improvements (see Strategy Log).

### Current champion (the ONLY anchor you target)
{anchor_block}

### Top-ranked paradigm descriptions (for context only — code withheld)
{inspiration_block}
{strategy_log_block}
## Your Task

A new paradigm will NOT help here — previous paradigm trials confirm
that. What WILL help is a **precise, structural improvement to the
champion**. Be the careful surgeon, not the wild inventor.

First, write a ``## Analysis`` section. Its first line MUST be
``MOVE: SURGICAL``. After that line, include exactly these four
sub-sections:

1. **Tightest constraint.** What is the single binding constraint
   limiting the champion's score *right now*? Cite the exact
   mechanism in the champion's code (function name, loop variable,
   magic constant). Examples: "the SA cooling schedule freezes at
   step≈6000, before the perimeter constraint fully relaxes", "the
   feasibility-repair step in `project()` shrinks circles uniformly,
   wasting per-circle slack".
2. **Structural fix.** Propose ONE structural change — not a
   constant tweak — that loosens that constraint. Examples: "add a
   per-circle slack budget before the global perimeter projection",
   "interleave a hex-grid restart every 5000 SA steps when no move
   was accepted in the last 500", "introduce a local Lloyd polish
   after every 1000 SA accepts".
3. **Preservation list.** A bullet list of all routines / constants
   that stay byte-for-byte identical. The fix must be local.
4. **Expected delta.** Your honest guess at how much score the fix
   buys, and why.

Then write the complete program. The overall algorithm class MUST
remain the champion's. Do not propose a fundamentally different
algorithm — synthesis and paradigm-shift modes exist for that. Do not
just retune constants — the mutation worker is doing that. The fix
must be **structural** and **local**.

### Critical requirements
1. Function signature MUST match exactly: `{function_signature}`
2. Include ALL necessary imports at the top of your code
3. Keep the champion's overall control flow and naming intact

{format_instruction}
{analysis_output_order_hint}\
"""


def _strategy_log_block(recent_trials: Sequence[str]) -> str:
    if not recent_trials:
        return ""
    return (
        "\n## Strategy Log (recent paradigm attempts)\n"
        + "\n".join(f"- {line}" for line in recent_trials)
        + "\n"
    )


def build_synthesis_prompt(
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
    """Synthesis mode: 2-3 anchors close in score, hybridise them."""
    return SYNTHESIS_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        n_evaluations=n_evaluations,
        n_cells=n_cells,
        stagnation=stagnation,
        anchor_block=_anchor_block(anchors),
        inspiration_block=_inspiration_block(inspirations),
        strategy_log_block=_strategy_log_block(recent_trials),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


def build_paradigm_shift_prompt(
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
    """Shift mode: 2 anchors, propose a genuinely new paradigm class."""
    return PARADIGM_SHIFT_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        n_evaluations=n_evaluations,
        n_cells=n_cells,
        stagnation=stagnation,
        anchor_block=_anchor_block(anchors),
        inspiration_block=_inspiration_block(inspirations),
        strategy_log_block=_strategy_log_block(recent_trials),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )


def build_surgical_exploit_prompt(
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
    """Surgical mode: 1 anchor (the champion), top descriptions as
    inspiration only. Frontier writes a structural local fix."""
    return SURGICAL_EXPLOIT_PROMPT.format(
        problem_description=problem_description,
        function_signature=function_signature,
        n_evaluations=n_evaluations,
        n_cells=n_cells,
        stagnation=stagnation,
        anchor_block=_anchor_block(anchors[:1]),
        inspiration_block=_inspiration_block(inspirations),
        strategy_log_block=_strategy_log_block(recent_trials),
        format_instruction=OUTPUT_FORMAT_INSTRUCTION,
        analysis_output_order_hint=ANALYSIS_OUTPUT_ORDER_HINT,
    )
