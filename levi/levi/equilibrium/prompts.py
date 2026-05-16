"""Prompts for Punctuated Equilibrium paradigm shift generation."""

import re
from typing import Optional


_BLUEPRINT_SECTION_HEADERS = ("DIAGNOSIS", "APPROACH", "INVARIANTS", "PSEUDOCODE")


def parse_blueprint(text: str) -> Optional[dict[str, str]]:
    """Extract the four blueprint sections from a heavy-model response.

    Returns a dict with keys ``diagnosis`` / ``approach`` / ``invariants`` /
    ``pseudocode`` (lowercase). Missing sections are left as empty strings.
    Returns None only when the response contains NONE of the expected
    headers — that's the signal to fall back to legacy paradigm-shift code
    generation rather than feeding garbage to the implementer prompts.

    The parser is whitespace- and case-insensitive on the headers and
    tolerates extra prose between sections.
    """
    if not text:
        return None

    upper = text.upper()
    if not any(h in upper for h in _BLUEPRINT_SECTION_HEADERS):
        return None

    # Split on section headers while preserving the header in each chunk.
    pattern = re.compile(
        r"^\s*(DIAGNOSIS|APPROACH|INVARIANTS|PSEUDOCODE)\s*:\s*",
        re.IGNORECASE | re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None

    sections: dict[str, str] = {h.lower(): "" for h in _BLUEPRINT_SECTION_HEADERS}
    for i, m in enumerate(matches):
        name = m.group(1).lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        # Strip code fences if the model snuck one in for PSEUDOCODE.
        body = re.sub(r"^```[a-zA-Z]*\n?", "", body)
        body = re.sub(r"\n?```$", "", body)
        sections[name] = body.strip()

    return sections

# Adaptive paradigm shift prompts keyed by budget stage.
# Early: explore radically different approaches.
# Mid: synthesize and recombine strengths from existing solutions.
# Late: targeted refinement of weak spots while preserving what works.

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

## Your Challenge: PARADIGM SHIFT

Analyze the representative solutions above and identify their core algorithmic paradigms.

Your goal is to engineer a **fundamentally different algorithmic approach** that explores untapped regions of the solution space.

### Analysis Steps:
1. **Identify current paradigms**: What algorithmic strategies do the existing solutions use? (e.g., greedy, graph-based, dynamic programming, heuristic search, brute-force with pruning, etc.)
2. **Find the gap**: What paradigms are NOT represented in the current solutions?
3. **Design a novel approach**: Synthesize a solution using a completely different conceptual framework and data structure strategy than those found in the examples

### Instructions:
1. Study the function signature carefully - match it EXACTLY
2. Actively avoid the core logic, heuristics, and search patterns used in the existing solutions
3. Design a solution using a COMPLETELY DIFFERENT strategy

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

## Your Challenge: SYNTHESIZE A STRONGER SOLUTION

The archive has been evolving and found several decent approaches. Your goal is to **combine the best ideas** from the existing solutions into a stronger hybrid.

### Analysis Steps:
1. **Identify strengths**: What does each solution do well? What cases does each handle effectively?
2. **Identify weaknesses**: Where does each solution fall short? What edge cases or scenarios cause poor performance?
3. **Synthesize**: Build a new solution that combines the strongest elements from multiple approaches while addressing their individual weaknesses

### Instructions:
1. Study the function signature carefully - match it EXACTLY
2. Borrow and adapt the best techniques from the existing solutions
3. Address weaknesses you observe in the current approaches
4. The result should meaningfully improve on the existing solutions, not just copy one of them

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

## Your Challenge: TARGETED IMPROVEMENT

The archive is mature. The solutions above represent well-evolved approaches. Your goal is to make a **focused, high-impact improvement** to the best-performing approach.

### Analysis Steps:
1. **Study the best solution carefully**: Understand exactly what it does and why
2. **Find the weak spot**: What specific scenarios, edge cases, or parameter ranges cause the best solution to lose points?
3. **Make a surgical fix**: Improve the handling of those weak cases without degrading performance on cases that already work well

### Instructions:
1. Study the function signature carefully - match it EXACTLY
2. Start from the logic of the highest-scoring solution
3. Make targeted changes to address its specific weaknesses
4. Preserve the core strengths — do NOT rewrite from scratch

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

# Budget stage thresholds
EARLY_THRESHOLD = 0.3
LATE_THRESHOLD = 0.6


def get_budget_stage(
    budget_progress: float,
    stagnation: float | None = None,
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
# Heavy-Light Synthesis (HLS) — Strategic Blueprint prompts.
#
# The heavy model produces a SHORT structured blueprint (~200-400 tokens)
# instead of full code. Light models then implement the blueprint in
# parallel; the blueprint also conditions a fraction of subsequent main-
# loop mutations through a TTL window. This keeps heavy-model spend
# focused on reasoning, not on boilerplate, and turns one-shot heavy
# guidance into a durable search-direction signal.
# ---------------------------------------------------------------------------


STRATEGIC_BLUEPRINT_PROMPT = """# Strategic Blueprint Request

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Archive Snapshot ({n_evaluations} evaluations, {n_regions} behavioural regions)

Below are the best-performing solutions from each region. Treat them as
*evidence about what the search has discovered so far*, not as code to
edit:

{representative_solutions}
{trajectory_block}
## Your Task: Strategic Blueprint (NOT code)

You are the strategist for an evolutionary search system. Light implementer
models will turn your blueprint into concrete Python on the next step. Your
job is to **decide the next algorithmic direction**, not to write code.

Write a blueprint with EXACTLY these four sections, each plain text:

DIAGNOSIS:
- 1-3 sentences identifying the dominant algorithmic family in the archive
  and the structural weakness that explains the plateau.

APPROACH:
- 4-8 sentences describing ONE concrete new algorithmic direction that is
  meaningfully different from what is already represented. Be specific
  about data structures, optimisation routines, and the order of operations.

INVARIANTS:
- A bulleted list (3-5 items) of correctness conditions / output-shape
  constraints / API contracts the implementation MUST preserve.

PSEUDOCODE:
- A short (5-20 lines) pseudocode sketch. Plain English with code-like
  structure is fine; this is a sketch, not Python.

Hard rules:
- Do NOT emit Python code, ```python``` fences, or imports.
- Do NOT propose minor parameter tweaks of an existing solution — propose
  a different algorithmic strategy.
- Stay under {max_words} words total. Brevity is the point.

## Output
Start your response with the literal word `DIAGNOSIS:` and follow the four-
section format above. No preamble.
"""


BLUEPRINT_IMPLEMENTATION_PROMPT = """# Implement Strategic Blueprint

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Strategic Blueprint (from the search strategist)

The blueprint below decides the algorithmic direction. Implement it
faithfully — do NOT replace the APPROACH with an unrelated algorithm.

{blueprint_text}

## Reference Solutions (for context only; do NOT just copy them)

{representative_solutions}

## Your Task

Write a complete Python implementation of the APPROACH described in the
blueprint. The PSEUDOCODE sketch shows the intended structure; the
INVARIANTS list MUST hold in your code.

- Match the function signature EXACTLY: `{function_signature}`
- Include all imports. Use standard libraries (numpy, math, heapq,
  collections, itertools, functools) and torch if useful.
- No placeholders, ellipses, or pseudocode in the final output.

## Output
Output ONLY complete, runnable Python code in a ```python``` block. No
explanation before or after.
"""


BLUEPRINT_VARIANT_PROMPT = """# Variant Implementation of Strategic Blueprint

## Problem
{problem_description}

## Function Signature
```python
{function_signature}
```

## Strategic Blueprint
{blueprint_text}

## Reference Implementation (already accepted, score {base_score:.17g})
```python
{base_code}
```

## Your Task

Produce a DIFFERENT implementation of the SAME blueprint. Keep the APPROACH
intact but vary:
- Data structure choices (array vs heap vs dict) where the blueprint is
  silent on the details.
- Secondary heuristics, tie-breaking rules, constants.
- Edge-case handling and early-exit conditions.
- Numerical precision or stability tactics.

Hard rules:
- Stay faithful to APPROACH and INVARIANTS.
- Match the function signature exactly: `{function_signature}`.
- Output complete, runnable Python.

## Output
Output ONLY complete Python code in a ```python``` block. No prose.
"""


VARIANT_GENERATION_PROMPT = """# Generate Variant of Paradigm Shift Solution

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

## Your Task
Generate a VARIANT of the above paradigm shift solution by:
1. Keeping the core algorithmic approach intact
2. Making targeted modifications to:
   - Constants and thresholds
   - Secondary heuristics
   - Edge case handling
   - Implementation details

The variant should explore nearby regions of the solution space while preserving the novel approach.

## Output
Output ONLY the complete Python code in a ```python block.
"""
