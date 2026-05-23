"""Tests for the AST structural signature used by the Pool second-pass dedup."""

from __future__ import annotations

from levi.simple.ast_signature import N_FEATURES, ast_cosine, compute_ast_signature


def test_signature_shape_and_dtype() -> None:
    sig = compute_ast_signature("x = 1\n")
    assert sig.shape == (N_FEATURES,)
    assert sig.dtype.name == "float32"


def test_parse_failure_returns_zero_vector() -> None:
    sig = compute_ast_signature("def !@#$ broken")
    assert sig.shape == (N_FEATURES,)
    assert (sig == 0).all()


def test_empty_string_returns_zero_vector() -> None:
    sig = compute_ast_signature("")
    assert (sig == 0).all()


def test_cosine_identical_code_is_one() -> None:
    code = "def f(x):\n    return sum(x)\n"
    sig_a = compute_ast_signature(code)
    sig_b = compute_ast_signature(code)
    assert ast_cosine(sig_a, sig_b) > 0.999


def test_cosine_distinct_structures_drops() -> None:
    # A tight loop with branches vs a one-liner comprehension differ in
    # loop_count, branch_count, comprehension_count, depth, …
    loop_code = (
        "def f(x):\n"
        "    total = 0\n"
        "    for i in range(len(x)):\n"
        "        if x[i] > 0:\n"
        "            total += x[i]\n"
        "    return total\n"
    )
    comp_code = "def f(x):\n    return sum(v for v in x if v > 0)\n"
    sig_loop = compute_ast_signature(loop_code)
    sig_comp = compute_ast_signature(comp_code)
    # Different but not orthogonal — both still look like Python functions.
    sim = ast_cosine(sig_loop, sig_comp)
    assert 0.5 < sim < 0.97, f"expected meaningful separation, got {sim}"


def test_cosine_handles_zero_vector() -> None:
    sig = compute_ast_signature("def f(): pass\n")
    zero = compute_ast_signature("def !!! syntax error")
    # ast_cosine with a zero-norm vector must return 0.0, never NaN.
    assert ast_cosine(sig, zero) == 0.0
    assert ast_cosine(zero, sig) == 0.0
