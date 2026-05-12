"""
Reflective Review — LLM-driven strategic re-plan when search stagnates.

Compared to AdaEvolve's paradigm breakthrough (which produces one idea for
one mutation step), the FORE review takes the *fertility map* of clusters
(effective / exhausted / embryonic) as input and returns a structured
plan that persists for a small window of generations.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from skydiscover.llm.llm_pool import LLMPool

logger = logging.getLogger(__name__)


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class FertilityReview:
    """Structured strategic plan produced by the reviewer."""

    effective_lineages: List[Dict[str, Any]] = field(default_factory=list)
    exhausted_lineages: List[Dict[str, Any]] = field(default_factory=list)
    embryonic_lineages: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)

    iteration_generated: int = 0
    uses_remaining: int = 3
    trigger_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "FertilityReview":
        if not isinstance(data, dict):
            return cls()
        keep = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in keep})

    def render_markdown(self) -> str:
        """Render the review for direct injection into the LLM prompt."""

        def _format_lineage_block(title: str, items: List[Dict[str, Any]]) -> str:
            if not items:
                return f"### {title}\n_(none)_\n"
            lines = [f"### {title}"]
            for it in items[:5]:
                label = it.get("label") or it.get("cluster_id") or "?"
                note = it.get("note", "")
                lines.append(f"- **{label}**: {note}")
            return "\n".join(lines) + "\n"

        sections = [
            "## REFLECTIVE REVIEW (active strategic plan)",
            f"_Trigger: {self.trigger_reason}; uses remaining: {self.uses_remaining}._",
            _format_lineage_block("Effective lineages (keep refining)", self.effective_lineages),
            _format_lineage_block("Exhausted lineages (avoid)", self.exhausted_lineages),
            _format_lineage_block("Embryonic lineages (worth exploring)", self.embryonic_lineages),
            "### Next steps",
        ]
        if self.next_steps:
            for i, step in enumerate(self.next_steps[:6], 1):
                sections.append(f"{i}. {step}")
        else:
            sections.append("_(none)_")
        return "\n".join(sections)


class ReflectiveReviewer:
    """Generates ``FertilityReview`` objects via an LLM call."""

    SYSTEM_HEADER = (
        "You are an expert algorithm researcher reviewing an ongoing program-search run.\n"
        "Your job is to read a fertility map of strategy clusters and produce a concise,\n"
        "actionable strategic plan that the search loop can follow for the next ~3 generations."
    )

    def __init__(
        self,
        llm_pool: LLMPool,
        system_message: str = "",
        evaluator_code: str = "",
        max_evaluator_chars: int = 4000,
    ):
        self.llm_pool = llm_pool
        self.system_message = system_message or ""
        self.evaluator_code = (evaluator_code or "")[:max_evaluator_chars]

    async def generate(
        self,
        fertility_summary: List[Dict[str, Any]],
        recent_attempts: List[Dict[str, Any]],
        global_best_score: Optional[float],
        trigger_reason: str = "stagnation",
        iteration: int = 0,
        timeout: float = 120.0,
    ) -> Optional[FertilityReview]:
        """Call the LLM and parse a ``FertilityReview``.

        Returns ``None`` on any failure; the caller is expected to fall back
        to vanilla sampling for that iteration.
        """
        user_msg = self._build_user_message(
            fertility_summary, recent_attempts, global_best_score, trigger_reason
        )
        system_msg = f"{self.SYSTEM_HEADER}\n\n# Task context\n{self.system_message.strip()}"

        try:
            result = await asyncio.wait_for(
                self.llm_pool.generate(system_msg, [{"role": "user", "content": user_msg}]),
                timeout=timeout,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("FORE: ReflectiveReviewer LLM call failed: %s", e)
            return None

        text = getattr(result, "text", None)
        if not text:
            logger.warning("FORE: ReflectiveReviewer got empty LLM response")
            return None

        review = self._parse(text)
        if review is None:
            logger.warning("FORE: ReflectiveReviewer could not parse JSON; raw=%r", text[:300])
            return None

        review.iteration_generated = iteration
        review.trigger_reason = trigger_reason
        return review

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_message(
        self,
        fertility_summary: List[Dict[str, Any]],
        recent_attempts: List[Dict[str, Any]],
        global_best_score: Optional[float],
        trigger_reason: str,
    ) -> str:
        lines: List[str] = []
        lines.append("# Fertility map (one row per strategy cluster)")
        if not fertility_summary:
            lines.append("_(no data — search just started)_")
        else:
            lines.append(
                "| cluster_id | label | size | mean_fitness | mean_delta_plus | negative_frac |"
            )
            lines.append("|---|---|---|---|---|---|")
            for row in fertility_summary[:15]:
                lines.append(
                    "| {cid} | {lab} | {sz} | {mf:.4f} | {mdp:.4f} | {nf:.2f} |".format(
                        cid=row.get("cluster_id", "?"),
                        lab=(row.get("label") or "?")[:40],
                        sz=row.get("size", 0),
                        mf=float(row.get("mean_fitness", 0.0) or 0.0),
                        mdp=float(row.get("mean_delta_plus", 0.0) or 0.0),
                        nf=float(row.get("negative_frac", 0.0) or 0.0),
                    )
                )

        lines.append("")
        lines.append("# Recent attempts (most recent last)")
        if not recent_attempts:
            lines.append("_(none)_")
        else:
            for att in recent_attempts[-10:]:
                lab = (att.get("strategy_label") or "?")[:40]
                verdict = att.get("verdict") or "?"
                fit = att.get("fitness")
                fit_str = f"{fit:.4f}" if isinstance(fit, (int, float)) else "?"
                desc = (att.get("description") or "").strip().splitlines()
                desc_one = desc[0] if desc else ""
                lines.append(
                    f"- [{verdict:>13}] {lab}: fitness={fit_str} — {desc_one[:140]}"
                )

        lines.append("")
        if global_best_score is not None:
            lines.append(f"Global best score so far: {global_best_score:.4f}")
        lines.append(f"Trigger reason: {trigger_reason}")

        if self.evaluator_code:
            lines.append("")
            lines.append("# Evaluator (excerpt)")
            lines.append("```")
            lines.append(self.evaluator_code)
            lines.append("```")

        lines.append("")
        lines.append("# Output")
        lines.append("Respond with a single JSON object (no prose around it) matching:")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "effective_lineages": [{"label": "<short>", "note": "<why keep>"}],
                    "exhausted_lineages": [{"label": "<short>", "note": "<why avoid>"}],
                    "embryonic_lineages": [
                        {"label": "<short>", "note": "<concrete direction to try>"}
                    ],
                    "next_steps": ["<concrete instruction 1>", "<concrete instruction 2>"],
                },
                indent=2,
            )
        )
        lines.append("```")
        lines.append("Keep each list at most 5 items. Be concrete and concise.")
        return "\n".join(lines)

    @staticmethod
    def _parse(text: str) -> Optional[FertilityReview]:
        # Strip code fences if any.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        # Try a direct parse, otherwise pull out the first {...} block.
        for candidate in (cleaned, _JSON_BLOCK_RE.search(cleaned).group(0) if _JSON_BLOCK_RE.search(cleaned) else None):
            if not candidate:
                continue
            try:
                data = json.loads(candidate)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue

            def _list_of_dicts(key: str) -> List[Dict[str, Any]]:
                val = data.get(key, []) or []
                if not isinstance(val, list):
                    return []
                out: List[Dict[str, Any]] = []
                for item in val:
                    if isinstance(item, dict):
                        out.append({k: v for k, v in item.items() if isinstance(k, str)})
                    elif isinstance(item, str):
                        out.append({"label": item, "note": ""})
                return out

            steps = data.get("next_steps", []) or []
            if not isinstance(steps, list):
                steps = []
            steps_clean: List[str] = []
            for s in steps:
                if isinstance(s, str):
                    steps_clean.append(s.strip()[:400])
                elif isinstance(s, dict):
                    txt = s.get("text") or s.get("step") or s.get("note") or ""
                    if isinstance(txt, str) and txt.strip():
                        steps_clean.append(txt.strip()[:400])

            return FertilityReview(
                effective_lineages=_list_of_dicts("effective_lineages"),
                exhausted_lineages=_list_of_dicts("exhausted_lineages"),
                embryonic_lineages=_list_of_dicts("embryonic_lineages"),
                next_steps=steps_clean,
            )
        return None
