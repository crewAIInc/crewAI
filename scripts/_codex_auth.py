#!/usr/bin/env python3
"""Helpers for detecting reusable local Codex authentication state."""

# ruff: noqa: S607

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def _codex_home() -> Path:
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()


def local_codex_auth_status() -> tuple[bool, str]:
    """Return whether local auth.json contains usable Codex credentials."""
    auth_json_path = _codex_home() / "auth.json"
    if not auth_json_path.exists():
        return False, f"codex auth.json not found at {auth_json_path}"

    try:
        payload = json.loads(auth_json_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return False, f"codex auth.json at {auth_json_path} is unreadable: {exc}"
    except UnicodeDecodeError as exc:
        return False, f"codex auth.json at {auth_json_path} is not valid UTF-8: {exc}"
    except json.JSONDecodeError as exc:
        return False, f"codex auth.json at {auth_json_path} is invalid: {exc.msg}"

    if not isinstance(payload, dict):
        return False, f"codex auth.json at {auth_json_path} is not a JSON object"

    tokens = payload.get("tokens")
    if isinstance(tokens, dict) and (
        tokens.get("access_token") or tokens.get("refresh_token")
    ):
        return True, f"codex auth.json loaded from {auth_json_path}"

    if payload.get("OPENAI_API_KEY"):
        return True, f"codex auth.json contains OPENAI_API_KEY at {auth_json_path}"

    return False, f"codex auth.json at {auth_json_path} does not contain usable credentials"


def codex_auth_status() -> tuple[bool, str]:
    """Return whether Codex OAuth state is available locally."""
    auth_json_available, auth_json_message = local_codex_auth_status()
    if auth_json_available:
        return True, auth_json_message

    try:
        proc = subprocess.run(
            ["codex", "login", "status"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        return False, auth_json_message
    except Exception as exc:  # noqa: BLE001
        return False, f"codex login status failed: {exc}"

    message = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    return proc.returncode == 0, (message or f"exit_code={proc.returncode}")
