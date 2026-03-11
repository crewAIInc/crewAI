#!/usr/bin/env python3
"""Minimal demo: call openai-codex/gpt-5.3-codex with prompt 'Hi'."""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import os
import sys

from crewai.llm import LLM
from _codex_auth import codex_auth_status, local_openai_api_key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="openai-codex/gpt-5.3-codex",
        help="Model in provider/model format.",
    )
    parser.add_argument(
        "--prompt",
        default="Hi",
        help="Prompt text.",
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
    args = parser.parse_args()

    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    logged_in, login_message = codex_auth_status()
    auth_json_api_key, auth_json_api_key_message = local_openai_api_key()

    if logged_in:
        # Default mode: prefer local Codex OAuth when signed in.
        os.environ["CREWAI_OPENAI_AUTH_MODE"] = "oauth_codex"
        # Ensure OAuth route is selected even if user has API key exported.
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_OAUTH_ACCESS_TOKEN", None)
        os.environ.pop("OPENAI_ACCESS_TOKEN", None)
        auth_mode = "oauth_codex"
        auth_strategy = "codex_oauth"
    else:
        # Not logged in: fall back to OpenAI API key route.
        os.environ.pop("CREWAI_OPENAI_AUTH_MODE", None)
        auth_mode = "api_key"
        auth_strategy = "api_key_fallback"
        if not os.getenv("OPENAI_API_KEY") and auth_json_api_key:
            os.environ["OPENAI_API_KEY"] = auth_json_api_key
            login_message = auth_json_api_key_message
        if not os.getenv("OPENAI_API_KEY"):
            print("auth_strategy=api_key_fallback")
            print(f"codex_login_status={login_message}")
            print("ERROR: not logged in to Codex OAuth and OPENAI_API_KEY is not set.")
            print("Set OPENAI_API_KEY, or run `codex login`, then retry.")
            return 2

    llm = LLM(
        model=args.model,
        api="responses",
        is_litellm=False,
        reasoning_effort=args.reasoning_effort,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    client_params = llm._get_client_params()
    auth_source = getattr(getattr(llm, "_resolved_openai_auth", None), "source", None)

    print(f"auth_strategy={auth_strategy}")
    print(f"auth_mode={auth_mode}")
    print(f"codex_login_status={login_message}")
    print(f"model={llm.model}")
    print(f"provider={llm.provider}")
    print(f"reasoning_effort={llm.reasoning_effort}")
    print(f"timeout={args.timeout}")
    print(f"max_retries={args.max_retries}")
    print(f"auth_source={auth_source}")
    print(f"base_url={client_params.get('base_url')}")

    result = llm.call(args.prompt)
    print("response:")
    print(str(result).strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
