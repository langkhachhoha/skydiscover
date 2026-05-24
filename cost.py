#!/usr/bin/env python3
"""Check remaining OpenRouter credits from the API key in .env."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

CREDITS_URL = "https://openrouter.ai/api/v1/credits"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _load_env() -> None:
    root = _repo_root()
    load_dotenv(root / ".env")
    load_dotenv(root / "levi" / ".env")


def _resolve_api_key() -> str:
    for name in ("OPENROUTER_API_KEY", "OPENROUTER_MANAGEMENT_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _mask_key(key: str) -> str:
    if len(key) <= 12:
        return "***"
    return f"{key[:8]}...{key[-4:]}"


def fetch_credits(api_key: str) -> dict:
    req = urllib.request.Request(
        CREDITS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


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
        payload = fetch_credits(api_key)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        print(f"OpenRouter tra ve loi HTTP {exc.code}.", file=sys.stderr)
        if body:
            print(body, file=sys.stderr)
        if exc.code in (401, 403):
            print(
                "\nGoi y: endpoint /credits can Management API key "
                "(https://openrouter.ai/settings/management-keys). "
                "Neu key hien tai la key goi model, hay them "
                "OPENROUTER_MANAGEMENT_API_KEY vao .env.",
                file=sys.stderr,
            )
        return 1
    except urllib.error.URLError as exc:
        print(f"Khong ket noi duoc OpenRouter: {exc.reason}", file=sys.stderr)
        return 1

    data = payload.get("data") or {}
    total_credits = float(data.get("total_credits", 0))
    total_usage = float(data.get("total_usage", 0))
    remaining = total_credits - total_usage

    print(f"Key: { _mask_key(api_key) }")
    print(f"Tong nap:        ${total_credits:.4f}")
    print(f"Da su dung:      ${total_usage:.4f}")
    print(f"Con lai:         ${remaining:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
