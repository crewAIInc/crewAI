#!/usr/bin/env python3
"""Check whether a model is callable with current local credentials."""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import os
import sys
import time

import openai
from crewai.llm import LLM
from _codex_auth import codex_auth_status, local_openai_api_key


def _targets_codex_model(model: str) -> bool:
    """Return whether model/provider string targets Codex backend."""
    normalized = model.strip().lower()
    return "codex" in normalized


def _configure_auth(model: str) -> tuple[str, str]:
    """Choose auth mode based on model family and local environment."""
    logged_in, login_message = codex_auth_status()
    auth_json_api_key, auth_json_api_key_message = local_openai_api_key()
    if _targets_codex_model(model) and logged_in:
        os.environ["CREWAI_OPENAI_AUTH_MODE"] = "oauth_codex"
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_OAUTH_ACCESS_TOKEN", None)
        os.environ.pop("OPENAI_ACCESS_TOKEN", None)
        return "codex_oauth", login_message

    os.environ.pop("CREWAI_OPENAI_AUTH_MODE", None)
    if os.getenv("OPENAI_API_KEY"):
        return "api_key", login_message
    if auth_json_api_key:
        os.environ["OPENAI_API_KEY"] = auth_json_api_key
        return "api_key", auth_json_api_key_message

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


def _run_single_check(
    model: str,
    api: str,
    prompt: str,
    reasoning_effort: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 4,
    max_attempts: int = 3,
) -> int:
    """Run one concrete model call check and print a normalized result."""
    for attempt in range(1, max_attempts + 1):
        try:
            llm = LLM(
                model=model,
                api=api,
                is_litellm=False,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
                max_retries=max_retries,
            )
            client_params = llm._get_client_params()
            auth_source = getattr(
                getattr(llm, "_resolved_openai_auth", None), "source", None
            )

            print(f"api={api}")
            print(f"attempt={attempt}")
            print(f"resolved_provider={llm.provider}")
            print(f"resolved_model={llm.model}")
            print(f"reasoning_effort={llm.reasoning_effort}")
            print(f"auth_source={auth_source}")
            print(f"base_url={client_params.get('base_url')}")
            print(f"timeout={timeout}")
            print(f"max_retries={max_retries}")

            result = llm.call(prompt)
            print("access_status=ok")
            print(f"response={str(result).strip()!r}")
            return 0
        except Exception as exc:  # noqa: BLE001
            status, reason = _classify_exception(exc)
            print(f"api={api}")
            print(f"attempt={attempt}")
            print(f"access_status={status}")
            print(f"reason={reason}")
            print(f"error_type={type(exc).__name__}")
            print(f"error={exc}")

            if status == "network_error" and attempt < max_attempts:
                retry_delay = attempt * 2
                print(f"retrying_in_seconds={retry_delay}")
                time.sleep(retry_delay)
                continue

            return 1

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
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Optional reasoning effort override passed to CrewAI/OpenAI.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Provider HTTP retry count passed to CrewAI/OpenAI.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="How many end-to-end attempts to make for transient network errors.",
    )
    args = parser.parse_args()

    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    auth_strategy, login_message = _configure_auth(args.model)

    print(f"requested_model={args.model}")
    print(f"api={args.api}")
    print(f"requested_reasoning_effort={args.reasoning_effort}")
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
        exit_codes.append(
            _run_single_check(
                args.model,
                api_name,
                args.prompt,
                reasoning_effort=args.reasoning_effort,
                timeout=args.timeout,
                max_retries=args.max_retries,
                max_attempts=args.max_attempts,
            )
        )
        print(f"check_end={api_name}")

    if all(code == 0 for code in exit_codes):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
