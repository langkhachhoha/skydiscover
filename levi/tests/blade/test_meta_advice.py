"""Tests for the BLADE meta-advisor prompt + error taxonomy.

The meta-advisor receives, on each trigger:

* The top-K archived programs (descriptions + scores).
* Recent admits as ``(source, score, delta_vs_parent)`` triples.
* A typed taxonomy of recent errors grouped by ``classify_error``.

These tests pin (a) that all three signal streams flow through the
prompt verbatim, (b) that the structured ``WORKING / TRY NEXT / AVOID``
contract is announced to the model, (c) that the error classifier maps
the real-world failure messages we observed in production to the right
buckets, and (d) that ``errors_only`` ablation parity is achievable by
passing empty success-side sequences.
"""

from __future__ import annotations

from levi.blade.prompts import (
    build_meta_advice_prompt,
    classify_error,
)


PROBLEM = "Maximise some objective f."
SIGNATURE = "def solve(x): ..."


# ---------------------------------------------------------------------------
# Error classifier
# ---------------------------------------------------------------------------


def test_classify_error_buckets_real_world_messages() -> None:
    """Every message below was observed in the production run log; the
    classifier must route each to its intended bucket so the advisor's
    taxonomy is meaningful."""
    cases: list[tuple[str, str]] = [
        # constraint
        ("Overlap between circles 0 and 2", "constraint"),
        ("Circles are not contained inside a rectangle of perimeter 4", "constraint"),
        ("Negative radius for circle 3", "constraint"),
        # shape mismatch
        ("operands could not be broadcast together with shapes (21,) (21,2)", "shape_mismatch"),
        ("array must be at least 2-d", "shape_mismatch"),
        ("too many indices for array: array is 2-dimensional, but 3 were indexed", "shape_mismatch"),
        # numpy api
        ("minimum() takes from 2 to 3 positional arguments but 4 were given", "numpy_api"),
        ("minimum() got an unexpected keyword argument 'initial'", "numpy_api"),
        # name / attr
        ("cannot access local variable 'ys' where it is not associated with a value", "name_or_attr"),
        ("'NoneType' object is not subscriptable", "name_or_attr"),
        # type error
        ("unsupported operand type(s) for -: 'float' and 'list'", "type_error"),
        ("too many values to unpack (expected 3)", "type_error"),
        # syntax
        ("Syntax error: invalid character '×' (U+00D7)", "syntax"),
        # timeout
        ("executor error: Process exceeded 600.0s timeout", "timeout"),
    ]
    misses: list[tuple[str, str, str]] = []
    for msg, expected in cases:
        got = classify_error(msg)
        if got != expected:
            misses.append((msg, expected, got))
    assert not misses, f"Misclassified: {misses}"


def test_classify_error_fallback() -> None:
    assert classify_error("") == "other"
    assert classify_error("totally unrelated gibberish kw9zXZ") == "other"


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_prompt_announces_three_section_contract() -> None:
    """The model must be told to emit WORKING / TRY NEXT / AVOID — this
    is the structural contract the orchestrator later parses and the
    operators consume verbatim."""
    p = build_meta_advice_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=1.23,
        n_evaluations=100,
        accept_rate=0.4,
        stagnation_level=0.2,
        top_descriptions=[("hex lattice", 1.23)],
        recent_admits=[("mutate_focused_fix", 1.23, 0.05)],
        errors_by_source=[("mutate_general", "Overlap between circles 0 and 2")],
        previous_advice=None,
    )
    assert "WORKING:" in p
    assert "TRY NEXT:" in p
    assert "AVOID:" in p
    # Order matters: WORKING < TRY NEXT < AVOID in the prompt body.
    assert p.index("WORKING:") < p.index("TRY NEXT:") < p.index("AVOID:")


def test_prompt_includes_all_signal_streams() -> None:
    p = build_meta_advice_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=2.5,
        n_evaluations=200,
        accept_rate=0.3,
        stagnation_level=0.6,
        top_descriptions=[
            ("hexagonal lattice with Adam relaxation", 2.50),
            ("simulated annealing with aspect-ratio sweeps", 2.45),
        ],
        recent_admits=[
            ("mutate_focused_fix", 2.50, 0.02),
            ("crossover_component_swap", 2.45, -0.01),
            ("mutate_targeted", 2.40, None),
        ],
        errors_by_source=[
            ("mutate_general", "Overlap between circles 1 and 2"),
            ("mutate_general", "Overlap between circles 3 and 5"),
            ("repair", "operands could not be broadcast together with shapes (21,) (21,2)"),
        ],
        previous_advice="prior wisdom about constraints",
    )
    # Top descriptions appear verbatim with score.
    assert "hexagonal lattice" in p
    assert "2.5000" in p or "2.5000" in p
    # Admits show source + delta sign.
    assert "mutate_focused_fix" in p
    assert "crossover_component_swap" in p
    assert "+0.0200" in p
    assert "-0.0100" in p
    # ``None`` delta renders as Δ=n/a.
    assert "Δ=n/a" in p
    # Error taxonomy buckets, with counts.
    assert "constraint ×2" in p
    assert "shape_mismatch ×1" in p
    # Per-source breakdown inside the bucket.
    assert "mutate_general×2" in p
    assert "repair×1" in p
    # Previous advice is carried over.
    assert "prior wisdom" in p


def test_prompt_handles_empty_signals_gracefully() -> None:
    """First trigger of a fresh run: archive is empty, no admits yet, no
    errors. Prompt must still render without raising."""
    p = build_meta_advice_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=float("-inf"),
        n_evaluations=0,
        accept_rate=0.0,
        stagnation_level=0.0,
        top_descriptions=[],
        recent_admits=[],
        errors_by_source=[],
        previous_advice=None,
    )
    assert "n/a" in p
    assert "(archive empty)" in p
    assert "(no admits in this window)" in p
    assert "(no failures in this window)" in p
    assert "(none yet" in p


def test_prompt_errors_only_mode_parity() -> None:
    """The ``meta_advice_mode='errors_only'`` ablation is implemented at
    the orchestrator layer by passing empty top_descriptions + empty
    recent_admits to this builder. Verify the prompt still renders and
    explicitly shows the empty-state placeholders so the model isn't
    confused into hallucinating successes."""
    p = build_meta_advice_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=2.5,
        n_evaluations=200,
        accept_rate=0.3,
        stagnation_level=0.6,
        top_descriptions=[],
        recent_admits=[],
        errors_by_source=[
            ("mutate_general", "Overlap between circles 1 and 2"),
        ],
        previous_advice=None,
    )
    assert "(archive empty)" in p
    assert "(no admits in this window)" in p
    # Errors still flow through.
    assert "constraint ×1" in p


def test_prompt_truncates_long_descriptions() -> None:
    long_desc = "x" * 1000
    p = build_meta_advice_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=1.0,
        n_evaluations=10,
        accept_rate=0.1,
        stagnation_level=0.0,
        top_descriptions=[(long_desc, 1.0)],
        recent_admits=[],
        errors_by_source=[],
        previous_advice=None,
    )
    # Truncation marker present.
    assert "…" in p
    # Not full 1000 chars repeated.
    assert "x" * 500 not in p


def test_prompt_truncates_long_error_messages() -> None:
    long_msg = "Overlap " + "x" * 1000
    p = build_meta_advice_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=1.0,
        n_evaluations=10,
        accept_rate=0.1,
        stagnation_level=0.0,
        top_descriptions=[],
        recent_admits=[],
        errors_by_source=[("mutate_general", long_msg)],
        previous_advice=None,
    )
    assert "…" in p
    # Bucket still constraint despite truncation.
    assert "constraint ×1" in p


def test_prompt_aggregates_repeated_errors_correctly() -> None:
    """20 overlap errors should collapse into a single ``constraint ×20``
    line — not 20 raw rows. This is the property that distinguishes the
    taxonomy from the old ``recent_errors=[5]`` list."""
    errors = [("mutate_focused_fix", f"Overlap between circles {i} and {i+1}") for i in range(20)]
    p = build_meta_advice_prompt(
        problem_description=PROBLEM,
        function_signature=SIGNATURE,
        best_score=1.0,
        n_evaluations=100,
        accept_rate=0.1,
        stagnation_level=0.0,
        top_descriptions=[],
        recent_admits=[],
        errors_by_source=errors,
        previous_advice=None,
    )
    assert "constraint ×20" in p
    assert "mutate_focused_fix×20" in p
