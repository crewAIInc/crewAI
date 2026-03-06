#!/usr/bin/env python3
"""Check whether a model is callable with current local credentials."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import openai
from crewai.llm import LLM


def _codex_login_status() -> tuple[bool, str]:
    """Return (logged_in, message) from `codex login status`."""
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
        return False, "codex CLI not found in PATH"
    except Exception as exc:  # noqa: BLE001
        return False, f"codex login status failed: {exc}"

    message = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    return proc.returncode == 0, (message or f"exit_code={proc.returncode}")


def _targets_codex_model(model: str) -> bool:
    """Return whether model/provider string targets Codex backend."""
    normalized = model.strip().lower()
    return "codex" in normalized


def _configure_auth(model: str) -> tuple[str, str]:
    """Choose auth mode based on model family and local environment."""
    logged_in, login_message = _codex_login_status()
    if _targets_codex_model(model) and logged_in:
        os.environ["CREWAI_OPENAI_AUTH_MODE"] = "oauth_codex"
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_OAUTH_ACCESS_TOKEN", None)
        os.environ.pop("OPENAI_ACCESS_TOKEN", None)
        return "codex_oauth", login_message

    os.environ.pop("CREWAI_OPENAI_AUTH_MODE", None)
    if os.getenv("OPENAI_API_KEY"):
        return "api_key", login_message

    if logged_in:
        return "codex_oauth_unusable_for_non_codex_model", login_message
    return "no_credentials", login_message


def _classify_exception(exc: Exception) -> tuple[str, str]:
    """Map provider exceptions to stable status codes."""
    if isinstance(exc, openai.NotFoundError):
        return "model_not_found", "Model does not exist or is not visible to this account."
    if isinstance(exc, openai.AuthenticationError):
        return "unauthorized", "Credential exists but lacks permission for this operation."
    if isinstance(exc, openai.RateLimitError):
        return "rate_limited", "Rate limit or quota exceeded."
    if isinstance(exc, openai.APIConnectionError):
        return "network_error", "Network or proxy error while calling the provider."
    if isinstance(exc, openai.BadRequestError):
        return "bad_request", "Request was rejected by the API."
    return "error", "Unhandled error."


def _run_single_check(model: str, api: str, prompt: str) -> int:
    """Run one concrete model call check and print a normalized result."""
    try:
        llm = LLM(model=model, api=api, is_litellm=False)
        client_params = llm._get_client_params()
        auth_source = getattr(getattr(llm, "_resolved_openai_auth", None), "source", None)

        print(f"api={api}")
        print(f"resolved_provider={llm.provider}")
        print(f"resolved_model={llm.model}")
        print(f"auth_source={auth_source}")
        print(f"base_url={client_params.get('base_url')}")

        result = llm.call(prompt)
        print("access_status=ok")
        print(f"response={str(result).strip()!r}")
        return 0
    except Exception as exc:  # noqa: BLE001
        status, reason = _classify_exception(exc)
        print(f"api={api}")
        print(f"access_status={status}")
        print(f"reason={reason}")
        print(f"error_type={type(exc).__name__}")
        print(f"error={exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Model in provider/model or bare model format.",
    )
    parser.add_argument(
        "--prompt",
        default="Reply with exactly: OK",
        help="Prompt used for the availability check.",
    )
    parser.add_argument(
        "--api",
        default="responses",
        choices=["responses", "chat", "both"],
        help="OpenAI API mode used by CrewAI.",
    )
    args = parser.parse_args()

    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    auth_strategy, login_message = _configure_auth(args.model)

    print(f"requested_model={args.model}")
    print(f"api={args.api}")
    print(f"auth_strategy={auth_strategy}")
    print(f"codex_login_status={login_message}")

    if auth_strategy == "no_credentials":
        print("access_status=blocked")
        print("reason=No Codex login and no OPENAI_API_KEY.")
        return 2

    if auth_strategy == "codex_oauth_unusable_for_non_codex_model":
        print("access_status=blocked")
        print("reason=Non-codex model requires OPENAI_API_KEY in this script.")
        return 2

    apis = ["responses", "chat"] if args.api == "both" else [args.api]
    exit_codes: list[int] = []
    for api_name in apis:
        print(f"check_start={api_name}")
        exit_codes.append(_run_single_check(args.model, api_name, args.prompt))
        print(f"check_end={api_name}")

    if any(code == 0 for code in exit_codes):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
