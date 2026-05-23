"""AST structural signature for second-pass Pool dedup.

The Pool's primary dedup uses *description* embeddings (semantic). When two
candidates produce near-identical descriptions but actually implement
structurally different code, the first pass would drop one of them and lose
real diversity. We compute a small, deterministic AST feature vector and let
the Pool admit the newcomer when its structural cosine sits below
``structural_cosine_threshold`` even if the description cosine is above the
niche threshold.

The features are intentionally cheap and bounded; on parse failure we return
a zero vector so the caller treats it as "no structural signal" and falls
back to description-only behavior.

Indexes (kept stable — Pool stores the vector and may compare across runs):

    0  ast_depth                 max depth of the AST
    1  cyclomatic_complexity     McCabe (1 + branches + bool-op fanout)
    2  loop_count                For / While nodes
    3  loop_nesting_max          deepest nested loop
    4  branch_count              If nodes
    5  function_def_count        FunctionDef nodes
    6  comprehension_count       ListComp / DictComp / SetComp / GeneratorExp
    7  call_count                Call nodes
    8  comparison_count          Compare nodes
    9  subscript_count           Subscript nodes (data-structure indexing)
    10 numeric_literal_count     int / float constants (parameter density)
    11 math_op_count             BinOp + UnaryOp nodes
    12 import_count              Import + ImportFrom nodes (lib mix)
    13 code_length               len(source) — a coarse size signal
"""

from __future__ import annotations

import ast
import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

N_FEATURES = 14


def compute_ast_signature(code: str) -> np.ndarray:
    """Return a length-``N_FEATURES`` vector for *code*.

    Returns a zero vector when the code cannot be parsed — callers treat
    that as "no structural signal" and skip the second-pass filter.
    """
    if not code:
        return np.zeros(N_FEATURES, dtype=np.float32)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.debug("ast_signature: parse failed; returning zero vector")
        return np.zeros(N_FEATURES, dtype=np.float32)

    ast_depth = _ast_depth(tree)
    cyclomatic = 1
    loop_count = 0
    branch_count = 0
    fn_def_count = 0
    comp_count = 0
    call_count = 0
    cmp_count = 0
    sub_count = 0
    num_count = 0
    math_op_count = 0
    import_count = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
            cyclomatic += 1
        elif isinstance(node, ast.BoolOp):
            cyclomatic += len(node.values) - 1
        if isinstance(node, (ast.For, ast.While)):
            loop_count += 1
        if isinstance(node, ast.If):
            branch_count += 1
        if isinstance(node, ast.FunctionDef):
            fn_def_count += 1
        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            comp_count += 1
        if isinstance(node, ast.Call):
            call_count += 1
        if isinstance(node, ast.Compare):
            cmp_count += 1
        if isinstance(node, ast.Subscript):
            sub_count += 1
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            num_count += 1
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            math_op_count += 1
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_count += 1

    loop_nesting = _max_loop_nesting(tree)

    vec = np.array(
        [
            ast_depth,
            cyclomatic,
            loop_count,
            loop_nesting,
            branch_count,
            fn_def_count,
            comp_count,
            call_count,
            cmp_count,
            sub_count,
            num_count,
            math_op_count,
            import_count,
            len(code),
        ],
        dtype=np.float32,
    )
    # Log-scale the long-tailed counts so a single 10× outlier (e.g. a
    # generated solution with 800 numeric literals) doesn't drown out the
    # cosine. Adding 1 keeps zero counts at zero.
    return np.log1p(vec)


def ast_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for AST signatures.

    Returns 0.0 if either vector is all-zero (parse failed or empty code) —
    that effectively skips the structural check and lets the Pool fall back
    to description-only dedup.
    """
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _ast_depth(tree: ast.AST) -> int:
    def _walk(node: ast.AST) -> int:
        children = list(ast.iter_child_nodes(node))
        if not children:
            return 1
        return 1 + max(_walk(c) for c in children)

    return _walk(tree)


def _max_loop_nesting(tree: ast.AST) -> int:
    def _walk(node: ast.AST, current: int) -> int:
        best = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While)):
                best = max(best, _walk(child, current + 1))
            else:
                best = max(best, _walk(child, current))
        return best

    return _walk(tree, 0)
