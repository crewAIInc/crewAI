#!/usr/bin/env python3
"""Validate strict OpenAI/Codex routing for gpt-5.2-codex and gpt-5.2-pro."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import random
import re
import time
from typing import Any, Callable

from openai import OpenAI

from crewai import Agent, Crew, Process, Task
from crewai.llms.providers.openai.completion import OpenAICompletion

PLATFORM_BASE_URL = "https://api.openai.com/v1"
CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


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
                detail = payload.get("detail")
                if isinstance(detail, str) and detail.strip():
                    message = sanitize(detail.strip())
        except Exception:
            pass

    return {
        "status": status,
        "type": err_type,
        "code": err_code,
        "message": message,
        "exception": type(exc).__name__,
    }


def is_retryable(info: dict[str, Any]) -> bool:
    status = info.get("status")
    msg = (info.get("message") or "").lower()
    if status is None and ("connection" in msg or "timeout" in msg or "tempor" in msg):
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
) -> tuple[bool, Any | None, dict[str, Any] | None]:
    for attempt in range(1, retries + 1):
        try:
            return True, fn(), None
        except Exception as exc:  # noqa: BLE001
            info = extract_error(exc)
            if attempt < retries and is_retryable(info):
                delay = base_sleep_seconds * (2 ** (attempt - 1)) + random.uniform(0.0, 0.3)
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


@contextmanager
def temp_env(*, set_values: dict[str, str] | None = None, unset_keys: list[str] | None = None):
    set_values = set_values or {}
    unset_keys = unset_keys or []
    old_values: dict[str, str | None] = {}
    for key in list(set_values.keys()) + list(unset_keys):
        old_values[key] = os.environ.get(key)

    try:
        for key in unset_keys:
            os.environ.pop(key, None)
        for key, value in set_values.items():
            os.environ[key] = value
        yield
    finally:
        for key, old in old_values.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def resolve_case_params(requested_model: str, route_mode: str) -> dict[str, Any]:
    if route_mode == "codex":
        env_set = {"CREWAI_OPENAI_AUTH_MODE": "oauth_codex"}
        env_unset = ["OPENAI_API_KEY", "OPENAI_OAUTH_ACCESS_TOKEN", "OPENAI_ACCESS_TOKEN"]
    else:
        env_set = {"CREWAI_OPENAI_AUTH_MODE": "oauth_codex"}
        env_unset = ["OPENAI_OAUTH_ACCESS_TOKEN", "OPENAI_ACCESS_TOKEN"]

    with temp_env(set_values=env_set, unset_keys=env_unset):
        llm = OpenAICompletion(model=requested_model, api="responses")
        client_params = llm._get_client_params()
        prepared = llm._prepare_responses_params(
            messages=[{"role": "user", "content": "route probe"}]
        )
        source = getattr(getattr(llm, "_resolved_openai_auth", None), "source", "unknown")
        base_url = client_params.get("base_url")
        api_key = client_params.get("api_key")
        effective_model = prepared.get("model", requested_model)
        return {
            "llm": llm,
            "source": source,
            "base_url": base_url,
            "api_key": api_key,
            "effective_model": effective_model,
            "api": llm.api,
        }


def run_sdk_case(requested_model: str, route_mode: str, prompt: str) -> tuple[bool, dict[str, Any]]:
    data = {
        "requested_model": requested_model,
        "effective_model": requested_model,
        "api": "responses",
        "base_url": None,
        "credential_source": "unknown",
        "status_code": None,
        "error_type": None,
        "error_code": None,
        "message": "",
        "pass": False,
    }

    try:
        resolved = resolve_case_params(requested_model, route_mode)
    except Exception as exc:  # noqa: BLE001
        info = extract_error(exc)
        data.update(
            {
                "status_code": info.get("status"),
                "error_type": info.get("type"),
                "error_code": info.get("code"),
                "message": info.get("message"),
            }
        )
        return False, data

    base_url = resolved["base_url"]
    api_key = resolved["api_key"]
    source = resolved["source"]
    effective_model = resolved["effective_model"]
    api = resolved["api"]

    data.update(
        {
            "effective_model": effective_model,
            "api": api,
            "base_url": base_url,
            "credential_source": source,
        }
    )

    if requested_model == "gpt-5.2-pro":
        if effective_model != requested_model:
            data["message"] = (
                f"requested_model ({requested_model}) != effective_model ({effective_model})"
            )
            return False, data
        if (base_url or "").rstrip("/") != PLATFORM_BASE_URL:
            data["message"] = f"gpt-5.2-pro must route to {PLATFORM_BASE_URL}, got {base_url}"
            return False, data
        if source in {"codex_auth_json_oauth", "codex_keyring_oauth", "env_oauth_access_token"}:
            data["message"] = (
                "gpt-5.2-pro requires Platform Responses API credential; "
                "codex OAuth access_token is not sufficient"
            )
            return False, data
    else:
        if (base_url or "").rstrip("/") != CODEX_BASE_URL:
            data["message"] = f"gpt-5.2-codex must route to {CODEX_BASE_URL}, got {base_url}"
            return False, data
        if source not in {"codex_auth_json_oauth", "codex_keyring_oauth"}:
            data["message"] = (
                "gpt-5.2-codex must use local Codex OAuth access_token source"
            )
            return False, data

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=0)

    def _call() -> Any:
        if requested_model == "gpt-5.2-codex":
            return client.responses.create(
                model=requested_model,
                input=[{"role": "user", "content": prompt}],
                instructions="You are a helpful assistant.",
                store=False,
                stream=True,
            )
        return client.responses.create(
            model=requested_model,
            input=prompt,
            instructions="You are a helpful assistant.",
            max_output_tokens=32,
        )

    ok, result, error = with_backoff(f"sdk.responses.{requested_model}", _call)
    if ok:
        if requested_model == "gpt-5.2-codex":
            text = ""
            for event in result:
                if getattr(event, "type", None) == "response.output_text.delta":
                    text += getattr(event, "delta", "") or ""
                if getattr(event, "type", None) == "response.completed":
                    break
            message = sanitize(text.strip())[:120]
        else:
            message = sanitize(getattr(result, "output_text", "") or str(result))[:120]
        data.update({"pass": True, "status_code": 200, "message": message})
        return True, data

    assert error is not None
    data.update(
        {
            "status_code": error.get("status"),
            "error_type": error.get("type"),
            "error_code": error.get("code"),
            "message": error.get("message"),
        }
    )
    return False, data


def run_crew_case(requested_model: str, route_mode: str, task_prompt: str) -> tuple[bool, dict[str, Any]]:
    data = {
        "requested_model": requested_model,
        "effective_model": requested_model,
        "api": "responses",
        "base_url": None,
        "credential_source": "unknown",
        "status_code": None,
        "error_type": None,
        "error_code": None,
        "message": "",
        "pass": False,
    }

    try:
        resolved = resolve_case_params(requested_model, route_mode)
    except Exception as exc:  # noqa: BLE001
        info = extract_error(exc)
        data.update(
            {
                "status_code": info.get("status"),
                "error_type": info.get("type"),
                "error_code": info.get("code"),
                "message": info.get("message"),
            }
        )
        return False, data

    base_url = resolved["base_url"]
    source = resolved["source"]
    effective_model = resolved["effective_model"]
    data.update(
        {
            "effective_model": effective_model,
            "api": resolved["api"],
            "base_url": base_url,
            "credential_source": source,
        }
    )

    def _kickoff() -> Any:
        if route_mode == "codex":
            env_set = {"CREWAI_OPENAI_AUTH_MODE": "oauth_codex"}
            env_unset = ["OPENAI_API_KEY", "OPENAI_OAUTH_ACCESS_TOKEN", "OPENAI_ACCESS_TOKEN"]
        else:
            env_set = {"CREWAI_OPENAI_AUTH_MODE": "oauth_codex"}
            env_unset = ["OPENAI_OAUTH_ACCESS_TOKEN", "OPENAI_ACCESS_TOKEN"]

        with temp_env(set_values=env_set, unset_keys=env_unset):
            llm = OpenAICompletion(
                model=requested_model,
                api="responses",
                instructions="You are a helpful assistant.",
            )
            agent = Agent(
                role="OAuthRoutingValidator",
                goal="Return a concise useful answer",
                backstory="Validating strict routing matrix",
                llm=llm,
                verbose=False,
            )
            task = Task(
                description=task_prompt,
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

    ok, result, error = with_backoff(f"crew.responses.{requested_model}", _kickoff)
    if ok:
        raw = getattr(result, "raw", None)
        text = sanitize(str(raw if raw is not None else result))
        data.update({"pass": True, "status_code": 200, "message": text[:120]})
        return True, data

    assert error is not None
    data.update(
        {
            "status_code": error.get("status"),
            "error_type": error.get("type"),
            "error_code": error.get("code"),
            "message": error.get("message"),
        }
    )
    return False, data


def print_case_result(prefix: str, result: dict[str, Any]) -> None:
    status = "PASS" if result["pass"] else "FAIL"
    print(
        f"{status} case={prefix} requested_model={result['requested_model']} "
        f"effective_model={result['effective_model']} api={result['api']} "
        f"base_url={result['base_url']} credential_source={result['credential_source']} "
        f"status_code={result['status_code']} error.type={result['error_type']} "
        f"error.code={result['error_code']} message={sanitize(result['message'])}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sdk", "crew", "all"], default="all")
    args = parser.parse_args()

    print("== ENV ==")
    print(f"OPENAI_API_KEY={'set' if os.getenv('OPENAI_API_KEY') else 'unset'}")
    print(f"CREWAI_OPENAI_AUTH_MODE={os.getenv('CREWAI_OPENAI_AUTH_MODE', '<unset>')}")

    sdk_ok = True
    crew_ok = True

    if args.mode in {"sdk", "all"}:
        print("== SDK ==")
        codex_ok, codex_result = run_sdk_case(
            requested_model="gpt-5.2-codex",
            route_mode="codex",
            prompt="print a python one-liner that prints OK",
        )
        print_case_result("sdk.responses.gpt-5.2-codex", codex_result)

        pro_ok, pro_result = run_sdk_case(
            requested_model="gpt-5.2-pro",
            route_mode="platform",
            prompt="Return exactly: OK",
        )
        print_case_result("sdk.responses.gpt-5.2-pro", pro_result)
        sdk_ok = codex_ok and pro_ok

    if args.mode in {"crew", "all"}:
        print("== CREW ==")
        codex_ok, codex_result = run_crew_case(
            requested_model="gpt-5.2-codex",
            route_mode="codex",
            task_prompt="Write one Python line that prints OK",
        )
        print_case_result("crew.responses.gpt-5.2-codex", codex_result)

        pro_ok, pro_result = run_crew_case(
            requested_model="gpt-5.2-pro",
            route_mode="platform",
            task_prompt="Respond with exactly: OK",
        )
        print_case_result("crew.responses.gpt-5.2-pro", pro_result)
        crew_ok = codex_ok and pro_ok

    overall = sdk_ok and crew_ok
    print(f"RESULT {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
