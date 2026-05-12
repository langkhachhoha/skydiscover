"""
FOREContextBuilder — injects FORE-specific guidance and the strategy-block
output requirement into the prompt.

Extends DefaultContextBuilder by:
- Adding a ``{fore_guidance}`` placeholder filled with (a) the active
  Reflective Review (if any), and (b) the parent's POV diagnostics.
- Adding a ``{fore_meta_instructions}`` placeholder describing exactly the
  ``<fore_meta>{...}</fore_meta>`` JSON block the LLM must append.

The parent-selection label injected by ``FOREDatabase.sample`` is rendered
by the base ``_format_current_program`` (because it sits in the parent_dict
key) so we don't need to duplicate that here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from skydiscover.config import Config
from skydiscover.context_builder.default import DefaultContextBuilder
from skydiscover.context_builder.utils import TemplateManager
from skydiscover.search.base_database import Program

logger = logging.getLogger(__name__)


class FOREContextBuilder(DefaultContextBuilder):
    """Prompt builder for FORE."""

    META_INSTRUCTIONS = (
        "After your code/diff above, append exactly one block of the form:\n\n"
        "<fore_meta>\n"
        "{\n"
        '  "strategy_label": "<short tag, kebab-case, e.g. \\"hexagonal-shell\\">",\n'
        '  "description": "<1-3 sentences on the algorithmic idea>",\n'
        '  "hypothesis": "<why this may be better, even if score may not show it immediately>",\n'
        '  "diff_from_parent": "<one line summary of the change vs the current solution>"\n'
        "}\n"
        "</fore_meta>\n\n"
        "Use plain JSON (no comments, no trailing commas). The block is parsed automatically;\n"
        "if you cannot fill a field, return an empty string for it but always emit the block."
    )

    def __init__(self, config: Config):
        super().__init__(config)
        default_dir = str(Path(__file__).parent.parent / "default" / "templates")
        fore_dir = str(Path(__file__).parent / "templates")
        self.template_manager = TemplateManager(default_dir, fore_dir, self.context_config.template_dir)

    def build_prompt(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        context = context or {}

        fore_guidance = self._build_fore_guidance(current_program, context)
        kwargs.pop("fore_guidance", None)
        kwargs.pop("fore_meta_instructions", None)

        return super().build_prompt(
            current_program,
            context,
            fore_guidance=fore_guidance,
            fore_meta_instructions=self.META_INSTRUCTIONS,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # FORE guidance assembly
    # ------------------------------------------------------------------

    def _build_fore_guidance(
        self,
        current_program: Union[Program, Dict[str, Program]],
        context: Dict[str, Any],
    ) -> str:
        sections: List[str] = []

        review_md = context.get("fore_review")
        if review_md:
            sections.append(str(review_md))

        diagnostics = context.get("fore_diagnostics")
        if diagnostics:
            sections.append(self._format_diagnostics(diagnostics))

        # Sibling verdicts for the parent — useful “what's been tried” hint.
        sibling_section = self._format_sibling_history(current_program)
        if sibling_section:
            sections.append(sibling_section)

        if not sections:
            return ""
        return "\n\n".join(sections)

    @staticmethod
    def _format_diagnostics(diagnostics: List[Dict[str, Any]]) -> str:
        if not diagnostics:
            return ""
        lines = [
            "## POV diagnostics (top candidates right now)",
            "| program | fitness | E[POV] | children (+/-) | mean Δ⁺ | cluster |",
            "|---|---|---|---|---|---|",
        ]
        for row in diagnostics:
            pid = str(row.get("program_id", "?"))[:8]
            lines.append(
                "| {pid} | {fit:.4f} | {pov:.4f} | {n}/{neg} | {mdp:.4f} | {cl} |".format(
                    pid=pid,
                    fit=float(row.get("fitness", 0.0) or 0.0),
                    pov=float(row.get("expected_pov", 0.0) or 0.0),
                    n=row.get("n", 0),
                    neg=row.get("neg", 0),
                    mdp=float(row.get("mean_delta_plus", 0.0) or 0.0),
                    cl=row.get("cluster", "?"),
                )
            )
        return "\n".join(lines)

    def _format_sibling_history(
        self,
        current_program: Union[Program, Dict[str, Program]],
    ) -> str:
        if isinstance(current_program, dict):
            if not current_program:
                return ""
            parent = list(current_program.values())[0]
        else:
            parent = current_program
        if parent is None:
            return ""

        # Walk the database for direct children of the parent (cheap enough at
        # small population sizes; FORE caps at ~80).
        db = getattr(self, "_db_cache", None) or getattr(self.config.search, "database", None)
        # The context builder doesn't get the database directly; siblings are
        # not strictly required here because the base builder already renders
        # `previous_attempts`. We keep this method as a placeholder for future
        # enhancement so FOREController can pass siblings via the context dict.
        return ""
