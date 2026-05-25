"""AST structural signature for second-pass Pool dedup.

Two implementations are provided and the active one is selected at call
time via the ``mode`` argument. Both return a fixed-length numeric vector
that downstream code consumes via :func:`ast_cosine`.

``"count14"`` (legacy): a length-14 vector of coarse counts (cyclomatic
complexity, loop count, comparison count, etc.) log-scaled with
``log1p``. Cheap and bounded, but in live runs it was observed to be far
too smooth: distinct paradigms (gradient descent vs simulated annealing)
routinely sat above cosine 0.97, which silently turned the structural
gate into a no-op and pushed all dedup pressure onto the description
embedding alone.

``"bigram"`` (default): a histogram of ``(parent_node_type, child_node_type)``
bigrams over the AST. The features are restricted to a stable, hand-picked
allowlist of node types (so the vector dimension stays constant across
runs and is comparable across snapshots) and L2-normalised so cosine is
the natural similarity. In live data the cross-paradigm cosine drops to
0.4-0.6 and same-paradigm paraphrases sit at 0.9+, giving the Pool a
meaningful structural signal again.

Both modes return zero vectors on parse failure — callers treat that as
"no structural signal" and fall back to description-only behavior.
"""

from __future__ import annotations

import ast
import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Legacy count-14 features (kept for the ablation study)
# ---------------------------------------------------------------------------

N_FEATURES_COUNT14 = 14

# ---------------------------------------------------------------------------
# Bigram features
# ---------------------------------------------------------------------------
#
# Node-type allowlist. Only nodes that materially distinguish paradigms are
# tracked — comments, formatted-value subnodes, type-annotation chrome, etc.
# are deliberately excluded so the histogram stays focused on control-flow
# and computation shape. Keep this list stable: it determines the feature
# axis order and changing it invalidates stored signatures.

_TRACKED_NODES: tuple[str, ...] = (
    # Control flow
    "FunctionDef", "AsyncFunctionDef", "Return", "If", "For", "While",
    "Break", "Continue", "Try", "ExceptHandler", "With",
    # Comprehensions
    "ListComp", "SetComp", "DictComp", "GeneratorExp", "comprehension",
    # Computation
    "Call", "BinOp", "UnaryOp", "BoolOp", "Compare", "Lambda",
    # Data access
    "Subscript", "Attribute", "Name", "Constant", "Tuple", "List",
    "Dict", "Set", "Slice", "Starred",
    # Assignment + scope
    "Assign", "AugAssign", "AnnAssign", "NamedExpr",
    # Imports
    "Import", "ImportFrom",
    # Top-level
    "Module", "Expr",
)

_NODE_IDX: dict[str, int] = {n: i for i, n in enumerate(_TRACKED_NODES)}
_N_NODES = len(_TRACKED_NODES)
N_FEATURES_BIGRAM = _N_NODES * _N_NODES

# Default mode (also used as the legacy import-time constant for callers
# that still reference ``N_FEATURES`` directly). We expose the bigram size
# because that is the production default and the only stable feature space
# going forward.
N_FEATURES = N_FEATURES_BIGRAM


AstMode = Literal["count14", "bigram"]


def compute_ast_signature(code: str, *, mode: AstMode = "bigram") -> np.ndarray:
    """Return a structural signature vector for *code*.

    Returns a zero vector when parsing fails — callers treat that as "no
    structural signal" and skip the structural gate.
    """
    if not code:
        return np.zeros(_dim_for(mode), dtype=np.float32)
    try:
        tree = ast.parse(code)
    except SyntaxError:
        logger.debug("ast_signature: parse failed; returning zero vector")
        return np.zeros(_dim_for(mode), dtype=np.float32)

    if mode == "count14":
        return _compute_count14(tree, code)
    return _compute_bigram(tree)


def _dim_for(mode: AstMode) -> int:
    return N_FEATURES_COUNT14 if mode == "count14" else N_FEATURES_BIGRAM


def _compute_bigram(tree: ast.AST) -> np.ndarray:
    """L2-normalised (parent, child) node-type histogram over an allowlist.

    Only bigrams where both ends are in :data:`_TRACKED_NODES` are counted.
    The matrix is flattened row-major. Normalisation lets the downstream
    cosine treat short and long programs comparably.
    """
    counts = np.zeros((_N_NODES, _N_NODES), dtype=np.float32)
    for parent in ast.walk(tree):
        parent_name = type(parent).__name__
        p_idx = _NODE_IDX.get(parent_name)
        if p_idx is None:
            continue
        for child in ast.iter_child_nodes(parent):
            c_idx = _NODE_IDX.get(type(child).__name__)
            if c_idx is None:
                continue
            counts[p_idx, c_idx] += 1.0
    vec = counts.reshape(-1)
    # log1p damps the long-tail of common bigrams (Name→Constant fires
    # everywhere; without dampening it dominates the cosine).
    vec = np.log1p(vec)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-9:
        vec = vec / norm
    return vec


def _compute_count14(tree: ast.AST, code: str) -> np.ndarray:
    """Legacy 14-count signature. Retained for the ablation study only."""
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
            ast_depth, cyclomatic, loop_count, loop_nesting, branch_count,
            fn_def_count, comp_count, call_count, cmp_count, sub_count,
            num_count, math_op_count, import_count, len(code),
        ],
        dtype=np.float32,
    )
    return np.log1p(vec)


def ast_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for AST signatures.

    Returns 0.0 if either vector is all-zero (parse failed or empty code)
    or if the two vectors have different dimensions (different modes), so
    the structural gate becomes a no-op rather than crashing.
    """
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    if a.shape != b.shape:
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
