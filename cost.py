#!/usr/bin/env python3
"""Check OpenRouter daily budget and account credits from API keys in .env."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

KEY_URL = "https://openrouter.ai/api/v1/key"
CREDITS_URL = "https://openrouter.ai/api/v1/credits"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_env() -> None:
    root = _repo_root()
    load_dotenv(root / ".env")
    load_dotenv(root / "levi" / ".env")


def _resolve_api_key() -> str:
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _resolve_management_key() -> str:
    return os.getenv("OPENROUTER_MANAGEMENT_API_KEY", "").strip()


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def _fetch_json(url: str, api_key: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "khong gioi han"
    return f"${value:.4f}"


def _print_key_budget(data: dict) -> None:
    limit = data.get("limit")
    limit_reset = data.get("limit_reset") or "khong reset"
    limit_remaining = data.get("limit_remaining")
    usage_daily = float(data.get("usage_daily", 0))

    print(f"Key:             {data.get('label', '(khong ro)')}")
    print(f"Gioi han:        {_fmt_money(limit)} ({limit_reset})")
    print(f"Da dung hom nay: ${usage_daily:.4f}")
    print(f"Con lai hom nay: {_fmt_money(limit_remaining)}")

    if limit is not None and limit_remaining is not None:
        used_pct = ((float(limit) - float(limit_remaining)) / float(limit)) * 100
        print(f"Da dung:         {used_pct:.1f}% gioi han")


def _print_account_credits(data: dict) -> None:
    total_credits = float(data.get("total_credits", 0))
    total_usage = float(data.get("total_usage", 0))
    remaining = total_credits - total_usage

    print()
    print("Tai khoan (tong):")
    print(f"  Tong nap:    ${total_credits:.4f}")
    print(f"  Da su dung:  ${total_usage:.4f}")
    print(f"  Con lai:     ${remaining:.4f}")


def main() -> int:
    _load_env()
    api_key = _resolve_api_key()
    if not api_key:
        print(
            "Khong tim thay API key. Dat OPENROUTER_API_KEY trong file .env.",
            file=sys.stderr,
        )
        return 1

    try:
        payload = _fetch_json(KEY_URL, api_key)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"OpenRouter tra ve loi HTTP {exc.code}.", file=sys.stderr)
        if body:
            print(body, file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"Khong ket noi duoc OpenRouter: {exc.reason}", file=sys.stderr)
        return 1

    data = payload.get("data") or {}
    print(f"API key: {_mask_key(api_key)}")
    print()
    _print_key_budget(data)

    management_key = _resolve_management_key()
    credits_key = management_key or api_key
    try:
        credits_payload = _fetch_json(CREDITS_URL, credits_key)
        _print_account_credits(credits_payload.get("data") or {})
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403) and not management_key:
            print()
            print(
                "Goi y: them OPENROUTER_MANAGEMENT_API_KEY vao .env de xem so du tai khoan.",
                file=sys.stderr,
            )
        elif exc.code not in (401, 403):
            body = exc.read().decode(errors="replace")
            print(f"\nKhong lay duoc so du tai khoan (HTTP {exc.code}).", file=sys.stderr)
            if body:
                print(body, file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
