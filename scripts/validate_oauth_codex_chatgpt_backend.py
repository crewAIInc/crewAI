#!/usr/bin/env python3
"""Validate CrewAI oauth_codex path against ChatGPT Codex backend.

This script intentionally avoids printing reusable secrets.
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from typing import Any, Callable

from openai import OpenAI

from crewai import Agent, Crew, Process, Task
from crewai.llms.providers.openai.completion import OpenAICompletion


def mask(value: str | None) -> str:
    if not value:
        return "<none>"
    value = value.strip()
    if not value:
        return "<none>"
    if len(value) <= 12:
        return f"len={len(value)} {value[:2]}...{value[-2:]}"
    return f"len={len(value)} {value[:6]}...{value[-4:]}"


def sanitize(text: str | None) -> str:
    if text is None:
        return ""
    out = str(text)
    out = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-<redacted>", out)
    out = re.sub(r"rt_[A-Za-z0-9_-]+", "rt_<redacted>", out)
    out = re.sub(
        r"eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}",
        "jwt.<redacted>",
        out,
    )
    return out


def extract_error(exc: Exception) -> dict[str, Any]:
    status = getattr(exc, "status_code", None)
    err_type = None
    err_code = None
    message = sanitize(str(exc))

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        payload = body.get("error") if isinstance(body.get("error"), dict) else body
        if isinstance(payload, dict):
            err_type = payload.get("type")
            err_code = payload.get("code")
            if payload.get("message"):
                message = sanitize(payload.get("message"))

    response = getattr(exc, "response", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
        try:
            payload = response.json()
            if isinstance(payload, dict):
                err_obj = (
                    payload.get("error")
                    if isinstance(payload.get("error"), dict)
                    else payload
                )
                if isinstance(err_obj, dict):
                    err_type = err_type or err_obj.get("type")
                    err_code = err_code or err_obj.get("code")
                    if err_obj.get("message"):
                        message = sanitize(err_obj.get("message"))
        except Exception:
            pass

    return {
        "status": status,
        "type": err_type,
        "code": err_code,
        "message": message,
        "exception": type(exc).__name__,
    }


def is_retryable(error_info: dict[str, Any]) -> bool:
    status = error_info.get("status")
    msg = (error_info.get("message") or "").lower()
    if status is None and (
        "connection" in msg or "timeout" in msg or "tempor" in msg
    ):
        return True
    if isinstance(status, int) and status >= 500:
        return True
    return False


def with_backoff(
    label: str,
    fn: Callable[[], Any],
    *,
    retries: int = 4,
    base_sleep_seconds: float = 1.0,
) -> tuple[bool, Any, dict[str, Any] | None]:
    for attempt in range(1, retries + 1):
        try:
            return True, fn(), None
        except Exception as exc:  # noqa: BLE001
            info = extract_error(exc)
            if attempt < retries and is_retryable(info):
                delay = base_sleep_seconds * (2 ** (attempt - 1)) + random.uniform(
                    0.0, 0.3
                )
                print(
                    f"RETRY label={label} attempt={attempt}/{retries} "
                    f"status={info['status']} delay={delay:.2f}s"
                )
                time.sleep(delay)
                continue
            return False, None, info
    return False, None, {
        "status": None,
        "type": None,
        "code": None,
        "message": f"{label} exhausted retries",
        "exception": "RuntimeError",
    }


def extract_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    raw = sanitize(str(response))
    return raw[:200]


def probe_client() -> tuple[OpenAICompletion, str, str, str]:
    probe = OpenAICompletion(model="gpt-5.2-pro", api="responses")
    params = probe._get_client_params()
    token = params.get("api_key")
    base_url = params.get("base_url")
    if not token:
        raise RuntimeError("No api_key/token resolved for oauth_codex probe")
    if not base_url:
        raise RuntimeError("No base_url resolved for oauth_codex probe")

    auth_source = getattr(getattr(probe, "_resolved_openai_auth", None), "source", None)
    print(f"auth.source={auth_source}")
    print(f"auth.token={mask(token)}")
    print(f"client.base_url={base_url}")

    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        raise RuntimeError(
            "oauth_codex resolved base_url must not end with /v1 for ChatGPT backend"
        )
    if "chatgpt.com/backend-api/codex" not in normalized:
        raise RuntimeError(
            "oauth_codex resolved base_url must target chatgpt.com/backend-api/codex"
        )

    return probe, token, base_url, auth_source


def run_sdk_checks(token: str, base_url: str) -> bool:
    print("== SDK CHECK ==")
    client = OpenAI(api_key=token, base_url=base_url, timeout=45.0, max_retries=0)

    requested_cases = [
        (
            "gpt-5.2-pro",
            "ping; respond with the single word OK",
        ),
        (
            "gpt-5.2-codex",
            "print a python one-liner that prints OK",
        ),
    ]

    all_passed = True
    for requested_model, prompt in requested_cases:
        backend_model = OpenAICompletion.CHATGPT_BACKEND_MODEL_ALIASES.get(
            requested_model, requested_model
        )
        label = f"sdk.responses.{requested_model}"

        def _call() -> Any:
            return client.responses.create(
                model=backend_model,
                input=[{"role": "user", "content": prompt}],
                instructions="You are a helpful assistant.",
                store=False,
                stream=True,
            )

        ok, result, error = with_backoff(label, _call)
        if ok:
            text = ""
            for event in result:
                if getattr(event, "type", None) == "response.output_text.delta":
                    text += getattr(event, "delta", "") or ""
                if getattr(event, "type", None) == "response.completed":
                    break
            text = sanitize(text.strip())
            print(
                f"PASS label={label} status=2xx requested_model={requested_model} "
                f"backend_model={backend_model} text={text[:120]!r}"
            )
        else:
            assert error is not None
            print(
                f"FAIL label={label} status={error['status']} error.type={error['type']} "
                f"error.code={error['code']} exception={error['exception']} "
                f"requested_model={requested_model} backend_model={backend_model} "
                f"message={sanitize(error['message'])}"
            )
            all_passed = False

    return all_passed


def run_crew_case(model: str, task_text: str) -> tuple[bool, str]:
    def _kickoff() -> Any:
        llm = OpenAICompletion(
            model=model,
            api="responses",
            instructions="You are a helpful assistant.",
        )
        agent = Agent(
            role="OAuthValidator",
            goal="Return concise outputs",
            backstory="Validating oauth_codex ChatGPT backend path",
            llm=llm,
            verbose=False,
        )
        task = Task(
            description=task_text,
            expected_output="A short answer",
            agent=agent,
        )
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        )
        return crew.kickoff()

    ok, result, error = with_backoff(f"crew.responses.{model}", _kickoff)
    if ok:
        raw = getattr(result, "raw", None)
        text = sanitize(str(raw if raw is not None else result))
        return True, text[:180]

    assert error is not None
    return (
        False,
        sanitize(
            f"status={error['status']} type={error['type']} code={error['code']} "
            f"exception={error['exception']} message={error['message']}"
        ),
    )


def run_crew_checks() -> bool:
    print("== CREW CHECK ==")
    cases = [
        ("gpt-5.2-pro", "Respond with exactly: OK"),
        ("gpt-5.2-codex", "Write one Python line that prints OK"),
    ]

    all_passed = True
    for model, prompt in cases:
        ok, detail = run_crew_case(model, prompt)
        if ok:
            print(f"PASS label=crew.responses.{model} detail={detail!r}")
        else:
            print(f"FAIL label=crew.responses.{model} detail={detail}")
            all_passed = False

    return all_passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["sdk", "crew", "all"],
        default="all",
        help="Validation mode to run",
    )
    args = parser.parse_args()

    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

    print("== ENV ==")
    print(f"CREWAI_OPENAI_AUTH_MODE={os.getenv('CREWAI_OPENAI_AUTH_MODE')}")
    print(f"OPENAI_API_KEY={'set' if os.getenv('OPENAI_API_KEY') else 'unset'}")
    print(
        "CREWAI_CODEX_CHATGPT_BASE_URL="
        f"{os.getenv('CREWAI_CODEX_CHATGPT_BASE_URL', '<default>')}"
    )

    try:
        _probe, token, base_url, auth_source = probe_client()
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL probe_failed={sanitize(type(exc).__name__ + ': ' + str(exc))}")
        return 2

    if auth_source not in {"codex_auth_json_oauth", "codex_keyring_oauth"}:
        print(f"FATAL unexpected_auth_source={auth_source}")
        return 3

    success = True
    if args.mode in {"sdk", "all"}:
        success = run_sdk_checks(token, base_url) and success

    if args.mode in {"crew", "all"}:
        success = run_crew_checks() and success

    print(f"RESULT {'PASS' if success else 'FAIL'}")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
