#!/usr/bin/env python3
"""Benchmark OpenRouter GPT-5 latency using a saved SkyDiscover prompt."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_OUTPUT_DIR = Path("outputs/local/evox_circle_packing_5_openrouter_20260512_codex")
DEFAULT_MODEL = "openai/gpt-5"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


def _iter_program_files(output_dir: Path) -> Iterable[Path]:
    checkpoints_dir = output_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []

    checkpoint_dirs = sorted(
        (p for p in checkpoints_dir.glob("checkpoint_*") if p.is_dir()),
        key=lambda p: int(p.name.rsplit("_", 1)[-1]) if p.name.rsplit("_", 1)[-1].isdigit() else -1,
        reverse=True,
    )

    files = []
    for checkpoint_dir in checkpoint_dirs:
        files.extend(sorted((checkpoint_dir / "programs").glob("*.json")))
    return files


def load_prompt_from_checkpoint(output_dir: Path, iteration: int) -> Tuple[str, str, Path]:
    """Return system/user prompt from the first program found for iteration."""
    for program_file in _iter_program_files(output_dir):
        with program_file.open("r", encoding="utf-8") as f:
            program = json.load(f)

        if program.get("iteration_found") != iteration:
            continue

        prompts = program.get("prompts") or {}
        if not prompts:
            continue

        prompt = prompts.get("diff_user_message") or next(iter(prompts.values()))
        system = prompt.get("system", "")
        user = prompt.get("user", "")
        if system and user:
            return system, user, program_file

    raise FileNotFoundError(
        f"No saved prompt found for iteration_found={iteration} under {output_dir}"
    )


def load_prompt_from_json(prompt_json: Path) -> Tuple[str, str, Path]:
    """Load a prompt JSON shaped as {'system': ..., 'user': ...} or EvoX prompts.json."""
    with prompt_json.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    if "system" in data and "user" in data:
        return data["system"], data["user"], prompt_json

    if "system_prompt" in data and "user_prompt" in data:
        return data["system_prompt"], data["user_prompt"], prompt_json

    raise ValueError(
        f"{prompt_json} must contain system/user or system_prompt/user_prompt fields"
    )


def call_once(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: Optional[int],
    timeout: float,
) -> Dict[str, Any]:
    start = time.perf_counter()
    params: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "extra_headers": {
            "HTTP-Referer": "https://github.com/skydiscover",
            "X-Title": "SkyDiscover OpenRouter Latency Benchmark",
        },
        "timeout": timeout,
        "reasoning_effort": "low",
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    response = client.chat.completions.create(**params)
    elapsed = time.perf_counter() - start
    text = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    return {
        "ok": True,
        "elapsed_sec": elapsed,
        "chars": len(text),
        "text": text,
        "reasoning": "",
        "usage": usage.model_dump() if usage else None,
        "finish_reason": response.choices[0].finish_reason,
    }


def _get_attr_or_extra(obj: Any, key: str) -> Any:
    value = getattr(obj, key, None)
    if value is not None:
        return value
    extra = getattr(obj, "model_extra", None) or {}
    if isinstance(extra, dict):
        return extra.get(key)
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.append(_as_text(item))
        return "".join(parts)
    if isinstance(value, dict):
        for key in ("summary", "text", "content"):
            if key in value:
                return _as_text(value[key])
        return ""
    return json.dumps(value, ensure_ascii=False)


def _extract_reasoning_delta(delta: Any) -> str:
    # OpenRouter can send both a simple text delta (reasoning/reasoning_content)
    # and OpenAI Responses-style details. Prefer the text delta to avoid
    # printing duplicate JSON-shaped metadata beside the actual words.
    for field in ("reasoning", "reasoning_content", "reasoning_text"):
        text = _as_text(_get_attr_or_extra(delta, field))
        if text:
            return text

    return _as_text(_get_attr_or_extra(delta, "reasoning_details"))


def _print_stream_chunk(section: str, text: str, current_section: Optional[str]) -> str:
    if current_section != section:
        print(f"\n===== STREAM {section.upper()} =====", flush=True)
        current_section = section
    print(text, end="", flush=True)
    return current_section


def call_once_stream(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: Optional[int],
    timeout: float,
) -> Dict[str, Any]:
    start = time.perf_counter()
    params: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "extra_headers": {
            "HTTP-Referer": "https://github.com/skydiscover",
            "X-Title": "SkyDiscover OpenRouter Latency Benchmark",
        },
        "timeout": timeout,
        "reasoning_effort": "low",
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens

    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = None
    usage = None
    current_section = None

    stream = client.chat.completions.create(**params)
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage

        if not getattr(chunk, "choices", None):
            continue

        choice = chunk.choices[0]
        finish_reason = getattr(choice, "finish_reason", None) or finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue

        reasoning_delta = _extract_reasoning_delta(delta)
        if reasoning_delta:
            reasoning_parts.append(reasoning_delta)
            current_section = _print_stream_chunk("reasoning", reasoning_delta, current_section)

        content_delta = getattr(delta, "content", None)
        if content_delta:
            text_parts.append(content_delta)
            current_section = _print_stream_chunk("output", content_delta, current_section)

    if current_section is not None:
        print("", flush=True)

    elapsed = time.perf_counter() - start
    text = "".join(text_parts)
    reasoning = "".join(reasoning_parts)
    if not reasoning:
        print("\n===== STREAM REASONING =====\n(no reasoning chunks received)", flush=True)
    if not text:
        print("\n===== STREAM OUTPUT =====\n(no output chunks received)", flush=True)

    return {
        "ok": True,
        "elapsed_sec": elapsed,
        "chars": len(text),
        "reasoning_chars": len(reasoning),
        "text": text,
        "reasoning": reasoning,
        "usage": usage.model_dump() if usage else None,
        "finish_reason": finish_reason,
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call openai/gpt-5 on OpenRouter repeatedly and report latency."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iteration", type=int, default=2)
    parser.add_argument("--prompt-json", type=Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=os.getenv("OPENROUTER_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")
    parser.add_argument(
        "--hide-input",
        action="store_true",
        help="Do not print the system/user prompt before calling the API.",
    )
    parser.add_argument(
        "--input-preview-chars",
        type=int,
        default=0,
        help="Truncate each input message to N chars. Default 0 prints full input.",
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY, API_KEY, or OPENAI_API_KEY")

    if args.prompt_json:
        system, user, prompt_source = load_prompt_from_json(args.prompt_json)
    else:
        system, user, prompt_source = load_prompt_from_checkpoint(args.output_dir, args.iteration)

    print(f"model={args.model}", flush=True)
    print(f"api_base={args.api_base}", flush=True)
    max_tokens_label = args.max_tokens if args.max_tokens is not None else "provider-default"
    print(f"runs={args.runs} max_tokens={max_tokens_label} timeout={args.timeout}", flush=True)
    print(f"stream={not args.no_stream}", flush=True)
    print(f"prompt_source={prompt_source}", flush=True)
    print(f"system_chars={len(system)} user_chars={len(user)}", flush=True)
    if not args.hide_input:
        system_to_print = system
        user_to_print = user
        if args.input_preview_chars > 0:
            system_to_print = system[: args.input_preview_chars]
            user_to_print = user[: args.input_preview_chars]
        print("\n===== INPUT SYSTEM =====", flush=True)
        print(system_to_print, flush=True)
        if args.input_preview_chars > 0 and len(system) > args.input_preview_chars:
            print(f"\n... truncated system input ({len(system)} chars total)", flush=True)
        print("\n===== INPUT USER =====", flush=True)
        print(user_to_print, flush=True)
        if args.input_preview_chars > 0 and len(user) > args.input_preview_chars:
            print(f"\n... truncated user input ({len(user)} chars total)", flush=True)

    client = OpenAI(api_key=api_key, base_url=args.api_base, timeout=args.timeout, max_retries=0)
    results = []

    for i in range(1, args.runs + 1):
        run_start = time.perf_counter()
        try:
            if args.no_stream:
                result = call_once(client, args.model, system, user, args.max_tokens, args.timeout)
            else:
                result = call_once_stream(
                    client, args.model, system, user, args.max_tokens, args.timeout
                )
        except Exception as exc:  # noqa: BLE001
            result = {
                "ok": False,
                "elapsed_sec": time.perf_counter() - run_start,
                "error": repr(exc),
            }
        results.append(result)

        if result["ok"]:
            print(
                f"run={i:02d} ok elapsed={result['elapsed_sec']:.2f}s "
                f"chars={result['chars']} reasoning_chars={result.get('reasoning_chars', 0)} "
                f"finish={result['finish_reason']} usage={result['usage']}",
                flush=True,
            )
            if result.get("reasoning"):
                print("\n===== FINAL REASONING =====", flush=True)
                print(result["reasoning"], flush=True)
            print("\n===== FINAL OUTPUT =====", flush=True)
            print(result.get("text", ""), flush=True)
        else:
            print(
                f"run={i:02d} error elapsed={result['elapsed_sec']:.2f}s "
                f"error={result['error']}",
                flush=True,
            )

    latencies = [r["elapsed_sec"] for r in results if r.get("ok")]
    errors = [r for r in results if not r.get("ok")]
    print("\nsummary", flush=True)
    print(f"ok={len(latencies)} error={len(errors)}", flush=True)
    if latencies:
        print(f"min={min(latencies):.2f}s", flush=True)
        print(f"mean={statistics.mean(latencies):.2f}s", flush=True)
        print(f"median={statistics.median(latencies):.2f}s", flush=True)
        print(f"p90={percentile(latencies, 90):.2f}s", flush=True)
        print(f"max={max(latencies):.2f}s", flush=True)

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
