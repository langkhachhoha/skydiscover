"""Per-LLM-call cost tracking, written to a JSONL sidecar.

Activated only when ``SKYDISCOVER_COST_LOG`` is set in the environment.
Each successful LLM call appends one record. A ``totals.json`` sibling file
is rewritten on every call so a runner can read the running total cheaply.

This exists because skydiscover's OpenAI wrapper normally discards the
``usage`` object from completion responses, and on a shared OpenRouter key
polling ``/credits`` cannot isolate this run from concurrent activity.

For OpenRouter requests we set ``extra_body={"usage": {"include": True}}``
so the provider returns ``usage.cost`` (USD) — that field is the ground
truth and is what gets logged. For non-OpenRouter providers we log
tokens but ``cost_usd`` will be ``None`` (no provider-side dollar number
is exposed).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

_LOCK = threading.Lock()
_TOTAL_COST = 0.0
_TOTAL_CALLS = 0
_TOTAL_PROMPT_TOKENS = 0
_TOTAL_COMPLETION_TOKENS = 0


def is_openrouter(api_base: Optional[str]) -> bool:
    return bool(api_base) and "openrouter.ai" in api_base.lower()


def cost_log_path() -> Optional[Path]:
    p = os.environ.get("SKYDISCOVER_COST_LOG")
    return Path(p) if p else None


def inject_openrouter_usage(params: Dict[str, Any], api_base: Optional[str]) -> None:
    """Mutate ``params`` so OpenRouter returns ``usage.cost`` in the response.

    No-op for non-OpenRouter endpoints. Safe to call multiple times; if the
    caller already provided ``extra_body['usage']`` we leave it alone.
    """
    if not is_openrouter(api_base):
        return
    extra = params.setdefault("extra_body", {})
    if not isinstance(extra, dict):
        return
    usage = extra.setdefault("usage", {"include": True})
    if isinstance(usage, dict) and "include" not in usage:
        usage["include"] = True


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _extract_usage_record(response: Any) -> Dict[str, Any]:
    usage = _get(response, "usage")
    if usage is None:
        return {}
    prompt = _get(usage, "prompt_tokens") or _get(usage, "input_tokens")
    completion = _get(usage, "completion_tokens") or _get(usage, "output_tokens")
    total = _get(usage, "total_tokens")
    cost = _get(usage, "cost")
    cost_details = _get(usage, "cost_details")
    prompt_details = _get(usage, "prompt_tokens_details")
    cached = _get(prompt_details, "cached_tokens") if prompt_details is not None else None
    rec: Dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cost_usd": float(cost) if cost is not None else None,
        "cached_tokens": cached,
    }
    if cost_details is not None:
        # Convert SDK objects to plain dict where possible.
        try:
            if hasattr(cost_details, "model_dump"):
                rec["cost_details"] = cost_details.model_dump()
            elif isinstance(cost_details, dict):
                rec["cost_details"] = cost_details
        except Exception:  # noqa: BLE001
            pass
    return rec


def record_call(
    response: Any,
    *,
    model: str,
    api_base: Optional[str],
    api_kind: str,
    call_site: str,
) -> None:
    """Append a record for one LLM response. No-op if SKYDISCOVER_COST_LOG unset."""
    log_path = cost_log_path()
    if log_path is None:
        return

    usage_rec = _extract_usage_record(response)
    record = {
        "ts": time.time(),
        "model": model,
        "api_base": api_base,
        "api_kind": api_kind,  # "chat.completions" | "responses"
        "call_site": call_site,  # e.g. "OpenAILLM._call_api"
        "response_id": _get(response, "id"),
        **usage_rec,
    }

    global _TOTAL_COST, _TOTAL_CALLS, _TOTAL_PROMPT_TOKENS, _TOTAL_COMPLETION_TOKENS
    with _LOCK:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        _TOTAL_CALLS += 1
        if record.get("cost_usd") is not None:
            _TOTAL_COST += float(record["cost_usd"])
        if record.get("prompt_tokens"):
            _TOTAL_PROMPT_TOKENS += int(record["prompt_tokens"])
        if record.get("completion_tokens"):
            _TOTAL_COMPLETION_TOKENS += int(record["completion_tokens"])

        totals = {
            "calls": _TOTAL_CALLS,
            "total_cost_usd": round(_TOTAL_COST, 6),
            "total_prompt_tokens": _TOTAL_PROMPT_TOKENS,
            "total_completion_tokens": _TOTAL_COMPLETION_TOKENS,
            "updated_ts": time.time(),
        }
        totals_path = log_path.with_suffix(".totals.json")
        tmp_path = totals_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(totals, indent=2))
        tmp_path.replace(totals_path)


def get_totals_snapshot() -> Dict[str, Any]:
    """Return a copy of the in-memory running totals."""
    with _LOCK:
        return {
            "calls": _TOTAL_CALLS,
            "total_cost_usd": round(_TOTAL_COST, 6),
            "total_prompt_tokens": _TOTAL_PROMPT_TOKENS,
            "total_completion_tokens": _TOTAL_COMPLETION_TOKENS,
        }


def format_cost_suffix() -> str:
    """Return a `[cost=$X.XXXX, calls=N]` suffix when tracking is on, else ``""``.

    Designed to be appended to iteration log lines so each iteration line
    carries the cumulative LLM spend at that point — making score-vs-cost
    curves derivable from log scrape alone.
    """
    if cost_log_path() is None:
        return ""
    with _LOCK:
        return f" [cost=${_TOTAL_COST:.4f}, llm_calls={_TOTAL_CALLS}]"
