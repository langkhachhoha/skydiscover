"""Prompts for Punctuated Equilibrium paradigm shift generation.

The three paradigm-shift prompts (``early``, ``mid``, ``late``) are dispatched
by :func:`get_budget_stage` keyed on the live PPS stagnation depth s(t):

  * ``early``  (s low)  — radically different algorithmic paradigm
  * ``mid``    (s med)  — synthesise strengths from existing solutions
  * ``late``   (s high) — surgical refinement of the best solution

All three accept a ``{strategy_log_block}`` placeholder so the strategy
history compiled by :mod:`levi.equilibrium.equilibrium` is injected before
the model writes its solution. The block is empty before any PE has fired.
"""

from typing import Optional


# Per-stage prompts. The {strategy_log_block} placeholder is filled by the
# adapter with a short summary of past paradigm-shift attempts (see
# StrategyRecord) so the heavy model knows which directions have already
# been tried and how they fared.
PARADIGM_SHIFT_PROMPTS = {
    "early": """# Algorithmic Paradigm Shift Challenge

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Current Best Solutions (From Different Behavioral Regions)

The archive has evolved through {n_evaluations} evaluations across {n_regions} behavioral regions.
Below are the best-performing solutions from each region:

{representative_solutions}
{strategy_log_block}
## Your Challenge: PARADIGM SHIFT (early-stage exploration)

We are still EARLY in the search. The archive only knows a handful of paradigms.
Your goal is to engineer a **fundamentally different algorithmic approach** that
explores untapped regions of the solution space.

### Analysis Steps:
1. **Identify current paradigms**: What algorithmic family does each solution above belong to? (greedy, graph search, dynamic programming, simulated annealing, gradient methods, brute-force with pruning, …)
2. **Find the gap**: Which paradigm classes are NOT represented in the archive or strategy log?
3. **Design a novel approach**: Pick ONE gap-paradigm and design a complete solution around it. The internal data structures, control flow, and termination condition must all reflect that paradigm — not a re-skin of an existing solution.

### Instructions:
1. Match the function signature exactly.
2. AVOID the core logic, heuristics, and search structures used in the archive.
3. AVOID any approach whose summary already appears in the Strategy Log (especially ones with non-positive deltas).
4. Pick a strategy that is structurally different, not just numerically retuned.

### Critical Requirements:
- Your function signature MUST match exactly: `{function_signature}`
- Use only standard Python libraries (numpy, collections, itertools, math, heapq, functools, etc.) and torch if needed
- The code must be syntactically valid and complete
- Include ALL necessary imports at the top
- Do NOT use placeholders, ellipses (...), or incomplete code
- Ensure the solution handles all edge cases

## Output
Output ONLY complete, runnable Python code in a ```python block. No explanations before or after.
""",
    "mid": """# Solution Synthesis Challenge

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Current Best Solutions (From Different Behavioral Regions)

The archive has evolved through {n_evaluations} evaluations across {n_regions} behavioral regions.
Below are the best-performing solutions from each region:

{representative_solutions}
{strategy_log_block}
## Your Challenge: SYNTHESISE A STRONGER SOLUTION (mid-stage consolidation)

Search has been running for a while and has accumulated several decent
approaches across distinct behavioural regions. Pure exploration is no longer
the best move — your goal is to **combine the best ideas** from the existing
solutions into a stronger hybrid that beats each individually.

### Analysis Steps:
1. **Per-region strengths**: For each region, identify *one* concrete mechanism it does well (e.g. better initialization, a clever tie-breaking rule, an aggressive prune).
2. **Per-region weaknesses**: For each region, identify *one* concrete failure mode (e.g. blows up at the boundary, ignores a constraint, gets stuck on adversarial inputs).
3. **Synthesis blueprint**: Choose 2-3 mechanisms to KEEP from different parents. Choose 1-2 weaknesses to FIX. State this implicitly via your code — do not write prose.

### Instructions:
1. Match the function signature exactly.
2. Borrow and adapt the strongest mechanisms from multiple solutions; do not just copy one of them.
3. Where the Strategy Log shows a synthesis that already produced no improvement, choose a different mechanism mix.
4. The result must be a structurally coherent program — not three solutions stitched together with `if/elif/else`.

### Critical Requirements:
- Your function signature MUST match exactly: `{function_signature}`
- Use only standard Python libraries (numpy, collections, itertools, math, heapq, functools, etc.) and torch if needed
- The code must be syntactically valid and complete
- Include ALL necessary imports at the top
- Do NOT use placeholders, ellipses (...), or incomplete code
- Ensure the solution handles all edge cases

## Output
Output ONLY complete, runnable Python code in a ```python block. No explanations before or after.
""",
    "late": """# Targeted Refinement Challenge

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Current Best Solutions (From Different Behavioral Regions)

The archive has evolved through {n_evaluations} evaluations across {n_regions} behavioral regions.
Below are the best-performing solutions from each region:

{representative_solutions}
{strategy_log_block}
## Your Challenge: TARGETED IMPROVEMENT (late-stage exploitation)

The archive is mature and stagnation is high. Radical rewrites at this stage
usually under-perform the best incumbent. Your goal is a **focused, high-impact
improvement** to the highest-scoring solution above.

### Analysis Steps:
1. **Study the best solution carefully**: Understand exactly what it does, and crucially WHERE it loses points (which inputs / which constraints / which edge cases).
2. **Find a single weak spot**: Pick ONE specific failure mode. Resist the urge to address several at once — those usually regress.
3. **Make a surgical fix**: Add a targeted patch (extra branch, post-processing step, tighter constraint check, refined tie-breaking) that fixes that failure WITHOUT touching the parts that already work.

### Instructions:
1. Match the function signature exactly.
2. Start from the logic of the highest-scoring solution; keep the overall control flow and data structures intact.
3. Do NOT rewrite the algorithm from scratch.
4. If the Strategy Log already shows a late-stage attempt with delta ≤ 0 targeting the same weak spot, choose a different weak spot.

### Critical Requirements:
- Your function signature MUST match exactly: `{function_signature}`
- Use only standard Python libraries (numpy, collections, itertools, math, heapq, functools, etc.) and torch if needed
- The code must be syntactically valid and complete
- Include ALL necessary imports at the top
- Do NOT use placeholders, ellipses (...), or incomplete code
- Ensure the solution handles all edge cases

## Output
Output ONLY complete, runnable Python code in a ```python block. No explanations before or after.
""",
}


# Budget stage thresholds (legacy — used only by callers that don't pass
# `stagnation`).
EARLY_THRESHOLD = 0.3
LATE_THRESHOLD = 0.6


def get_budget_stage(
    budget_progress: float,
    stagnation: Optional[float] = None,
    mid_threshold: float = 0.3,
    late_threshold: float = 0.7,
) -> str:
    """Map (budget_progress, stagnation) to a prompt-stage name.

    When `stagnation` is None we keep historical behaviour (always 'early').
    When provided, we route on the stagnation depth s(t) ∈ [0,1]:

      s < mid_threshold   → 'early'   (radical paradigm shift)
      s < late_threshold  → 'mid'     (synthesise strengths)
      s ≥ late_threshold  → 'late'    (surgical fix on best solution)
    """
    if stagnation is None:
        return "early"
    if stagnation < mid_threshold:
        return "early"
    if stagnation < late_threshold:
        return "mid"
    return "late"


# Default prompt (backwards compat) — same as early stage
PARADIGM_SHIFT_PROMPT = PARADIGM_SHIFT_PROMPTS["early"]


# ---------------------------------------------------------------------------
# Variants of an accepted paradigm shift (light-model "explore around" prompt)
# ---------------------------------------------------------------------------


VARIANT_GENERATION_PROMPT = """# Generate Variant {variant_idx} of {n_variants} — Paradigm Shift Solution

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Base Paradigm Shift Solution (Score: {base_score:.17g})
```python
{base_code}
```

## Your Variant Directive (variant {variant_idx} of {n_variants})
{variant_directive}

## Your Task
You are generating **variant {variant_idx} of {n_variants}** — a sibling fanout
from the same base solution. The other variants will explore DIFFERENT
directives; your job is to commit fully to the directive above so that the
sibling set spans different parts of the solution space.

Rules:
1. Keep the base's high-level paradigm class recognisable, but the directive
   above takes priority — if it asks you to replace a component, REPLACE it,
   don't just retune its constants.
2. Make at least one **non-cosmetic, structural** change consistent with the
   directive. Renaming variables, changing a single constant by < 2x, or
   adding a no-op safety check does NOT count.
3. Do NOT duplicate what the base already does well. If the directive
   conflicts with the base in some part, follow the directive there.

### Critical Requirements:
- Your function signature MUST match exactly: `{function_signature}`
- Use only standard Python libraries (numpy, collections, itertools, math, heapq, functools, etc.) and torch if needed
- The code must be syntactically valid and complete
- Include ALL necessary imports at the top
- Do NOT use placeholders, ellipses (...), or incomplete code
- Ensure the solution handles all edge cases

## Output
Output ONLY complete, runnable Python code in a ```python block. No explanations before or after.
"""


# Diversified exploration directives — orchestrator round-robins through this
# list when fanning out ``n_paradigm_variants`` siblings from the same base
# paradigm seed. The goal is to force each sibling into a distinct corner of
# the solution space instead of having every variant tweak the same handful
# of constants (which was the previous failure mode).
VARIANT_DIRECTIVES: tuple[str, ...] = (
    # 0: initialisation / starting point
    "**Re-design the INITIALISATION / starting state.** Replace how the base "
    "constructs its initial candidate (seed pattern, point layout, parameter "
    "init, warm-start). Keep the post-init optimisation pipeline intact, but "
    "feed it a structurally different starting point (e.g. switch from random "
    "to structured / lattice / problem-specific construction, or vice versa).",
    # 1: optimisation step rule
    "**Replace the OPTIMISATION STEP RULE / inner update.** Keep the overall "
    "loop structure but swap the update rule (e.g. gradient → coordinate "
    "descent, Adam → SGD-with-momentum, projected gradient → augmented "
    "Lagrangian, greedy → simulated annealing acceptance). The before/after "
    "of one iteration must look algorithmically different.",
    # 2: objective / loss shaping
    "**Re-shape the OBJECTIVE / loss the inner loop optimises.** Add or remove "
    "penalty / barrier / regulariser terms; change the smoothing/softmax "
    "temperature; switch between log-barrier and quadratic penalty for "
    "constraints. The final scoring is unchanged, but the surface the inner "
    "loop climbs should be different.",
    # 3: symmetry / structural prior
    "**Inject or break a SYMMETRY / structural prior.** If the base is "
    "unstructured, impose a symmetry (reflective, rotational, lattice, "
    "block-diagonal, hierarchical) and parametrise inside it. If the base "
    "already assumes a symmetry, break it intentionally and parametrise the "
    "broken degrees of freedom.",
    # 4: termination / restart policy
    "**Change the TERMINATION & RESTART policy.** Add multi-restart with "
    "best-of-K selection, basin hopping, or population-style retries; or "
    "conversely, replace early stopping with a longer single run. Adjust how "
    "the inner loop decides it's done so the outer behaviour is qualitatively "
    "different — not just a different number of iterations.",
    # 5: scale / discretisation / resolution
    "**Change the SCALE / DISCRETISATION / resolution.** Coarsen-then-refine "
    "(multi-resolution / multi-grid), or switch from continuous parameters to "
    "a discrete combinatorial search and back, or change problem size N for "
    "intermediate sub-problems before assembling the final answer.",
    # 6: hybrid composition
    "**Compose a HYBRID with a complementary technique.** Wrap the base in "
    "an outer loop that calls a fundamentally different solver for one stage "
    "(e.g. LP / SDP relaxation → rounding, branch-and-bound on a small "
    "subset, local search post-processor on the base's output). The base "
    "becomes one component of a pipeline, not the whole pipeline.",
    # 7: aggressive constant / shape rebalance
    "**Aggressively REBALANCE the hyperparameter regime.** Change at least "
    "two key hyperparameters by ≥ 3× simultaneously (learning rate, num "
    "iterations, batch / population size, temperature schedule, penalty "
    "weights). Pick a clearly different operating point — large-step "
    "exploration, or tiny-step deep refinement — not a small perturbation.",
)


# ---------------------------------------------------------------------------
# Strategy summarisation — light-model post-mortem after each PE event
# ---------------------------------------------------------------------------

STRATEGY_SUMMARY_PROMPT = """# Strategy Summary

Extract the core strategy used by the Python solution below. The summary will
be used by a later paradigm-shift step to avoid repeating weak ideas and to
recognize genuinely different algorithmic directions.

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Solution
```python
{code}
```

## Task
Identify:

- the algorithmic family, not just surface implementation details;
- the main tactic that makes this solution distinct for this problem;
- the input shapes, constraints, or cases where the tactic is likely to help;
- the specific reason it may lose points.

## Output — EXACTLY two lines, no preamble, no extra labels

IDEA: <one sentence naming the algorithmic family plus the concrete tactic used here>
QUALITY: <one sentence naming where it should score well and the most likely failure mode>

Hard rules:
- Total output <= 150 words.
- Be problem-specific: refer to constraints, data structure choices, pruning, ordering, approximation, search state, or objective handling when relevant.
- Avoid vague words such as "efficient", "optimized", "robust", "simple", "good", or "fast" unless paired with a concrete mechanism.
- Do not mention code style, imports, syntax, or implementation cleanliness.
"""


# ---------------------------------------------------------------------------
# Code Error Repair — one-shot light-model fix for broken candidates
# ---------------------------------------------------------------------------


CODE_REPAIR_PROMPT = """# Code Repair

You are repairing a buggy Python candidate. Produce the smallest complete fix
that makes the solution run correctly for the reported error while preserving
the candidate's algorithmic approach.

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Buggy Solution (parent score: {parent_score:.4g})
```python
{broken_code}
```

## Error
{error_msg}

## Rules
- Match the function signature exactly.
- Preserve the same algorithm, data structures, control flow shape, and heuristic intent.
- Fix the reported failure directly: syntax errors, missing imports, undefined names, bad variable scope, type mismatches, indexing/key errors, unpacking errors, or invalid return shape.
- Add helper functions only when needed to make the existing approach runnable.
- Include every import required by the final code.
- Keep changes conservative; do not replace the solution with a new paradigm.
- Handle empty, minimal, and boundary inputs implied by the problem when they are related to the reported failure.
- Do not add explanations, comments about the repair, placeholders, ellipses, pseudocode, or incomplete branches.

## Output
Output ONLY complete, runnable Python code in a ```python block. No explanations before or after.
"""
