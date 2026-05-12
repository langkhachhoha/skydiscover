"""
Strategy description parsing and per-program FORE metadata schema.

We ask the LLM to append a small JSON block in ``<fore_meta>...</fore_meta>``
tags. If the block is missing or malformed, we silently fall back to an empty
``StrategyDescription`` so a parse failure never breaks an iteration.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


FORE_BLOCK_RE = re.compile(r"<fore_meta>(.*?)</fore_meta>", re.DOTALL | re.IGNORECASE)


@dataclass
class StrategyDescription:
    """First-class strategy description carried on every candidate."""

    strategy_label: str = "unspecified"
    description: str = ""
    hypothesis: str = ""
    diff_from_parent: str = ""
    verdict: Optional[str] = None
    cluster_id: int = -1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "StrategyDescription":
        if not isinstance(data, dict):
            return cls()
        keep = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in keep})


def parse_strategy_block(llm_response: Optional[str]) -> StrategyDescription:
    """Extract the ``<fore_meta>`` JSON block from an LLM response.

    Returns an empty ``StrategyDescription`` on any failure.
    """
    if not llm_response:
        return StrategyDescription()
    m = FORE_BLOCK_RE.search(llm_response)
    if not m:
        return StrategyDescription()

    raw = m.group(1).strip()
    # Some models wrap JSON in code fences; strip them.
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)

    try:
        data = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        logger.debug("FORE: failed to parse <fore_meta> JSON (%s); using empty", e)
        return StrategyDescription()

    if not isinstance(data, dict):
        return StrategyDescription()

    def _s(key: str) -> str:
        val = data.get(key, "")
        if val is None:
            return ""
        if not isinstance(val, str):
            val = str(val)
        return val.strip()[:1000]

    return StrategyDescription(
        strategy_label=_s("strategy_label") or "unspecified",
        description=_s("description"),
        hypothesis=_s("hypothesis"),
        diff_from_parent=_s("diff_from_parent"),
    )


def compute_verdict(
    parent_fitness: Optional[float],
    child_fitness: Optional[float],
    parent_mean_delta_plus: float,
    threshold: float = 0.005,
) -> str:
    """Bucket the child's outcome into one of four verdicts.

    - ``improved``      : child > parent + threshold.
    - ``regressed``     : child < parent - threshold AND parent already had
                           a meaningful positive track record (penalised more).
    - ``stepping_stone``: roughly flat or modestly below parent — useful for
                           later mutations even if not a direct win.
    - ``dead_end``      : same as regressed but on a parent with no track
                           record (purely diagnostic for the LLM).
    """
    if parent_fitness is None or child_fitness is None:
        return "stepping_stone"
    delta = child_fitness - parent_fitness
    if delta > threshold:
        return "improved"
    if delta < -threshold:
        if parent_mean_delta_plus > 2 * threshold:
            return "regressed"
        return "dead_end"
    return "stepping_stone"


# ---------------------------------------------------------------------------
# Lightweight token-set clustering (no external embeddings).
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]+")


def tokenize_strategy(strategy: StrategyDescription) -> set:
    """Tokenize the strategy's textual fields for Jaccard-based clustering.

    Falls back to an empty set for unspecified descriptions; the caller is
    expected to either send the program to its own singleton cluster or
    re-tokenize with the code as a backstop.
    """
    parts = [
        strategy.strategy_label,
        strategy.description,
        strategy.diff_from_parent,
    ]
    text = " ".join(p for p in parts if p)
    tokens = {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 3}
    return tokens


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    if union == 0:
        return 1.0
    return inter / union
