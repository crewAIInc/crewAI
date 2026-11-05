from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
from uuid import uuid4

from crewai_core import plus_api
from dotenv import load_dotenv
import httpx
import pytest

from crewai.telemetry.tracing.grants import TraceGrantClient


@pytest.mark.parametrize(
    "machine",
    [
        {},
        {
            "CREWAI_PLUS_URL": "https://trusted.example",
            "CREWAI_USER_PAT": "shell-pat",
            "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN": "shell-integration",
            "CREWAI_TRACING_ENABLED": "false",
        },
    ],
)
def test_project_env_cannot_choose_amp_credential_or_tracing_consent(
    monkeypatch, tmp_path, machine
):
    from crewai_cli.run_crew import load_project_env

    monkeypatch.setattr(plus_api, "_MACHINE_AMP_ENV", machine)
    for name in plus_api._AMP_ENV_NAMES:
        monkeypatch.setenv(name, "previous-value")
    env = tmp_path / ".env"
    env.write_text(
        "CREWAI_PLUS_URL=https://attacker.example\n"
        "CREWAI_USER_PAT=project-pat\n"
        "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN=project-integration\n"
        "CREWAI_TRACING_ENABLED=true\n"
        "OPENAI_API_KEY=project-model-key\n"
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_project_env(env)
    load_dotenv(env, override=False)
    for name in plus_api._AMP_ENV_NAMES:
        assert os.environ[name] == machine.get(name, "")
    assert os.environ["OPENAI_API_KEY"] == "project-model-key"


def test_trace_grant_sends_login_only_to_machine_amp_and_rejects_foreign_viewer(
    monkeypatch,
):
    execution = str(uuid4())
    seen = []
    monkeypatch.setattr(
        plus_api, "_MACHINE_AMP_ENV", {"CREWAI_PLUS_URL": "https://trusted.example"}
    )
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://attacker.example")

    def respond(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "tier": "authenticated",
                "execution_uuid": execution,
                "token": "execution-scoped-token",
                "collector_url": "https://wharf.example/v1/traces",
                "expires_at": (
                    datetime.now(timezone.utc) + timedelta(minutes=5)
                ).isoformat(),
                "trace_url": "https://attacker.example/viewer",
            },
        )

    @contextmanager
    def client(verify=True):
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            yield http

    monkeypatch.setattr(plus_api.PlusAPI, "_http_client", staticmethod(client))
    grant = TraceGrantClient("saved-login").create(execution)
    assert len(seen) == 1
    assert (
        str(seen[0].url) == "https://trusted.example/crewai_plus/api/v1/tracing/grants"
    )
    assert seen[0].headers["Authorization"] == "Bearer saved-login"
    assert grant.trace_url is None
    assert grant.collector_url == "https://wharf.example/v1/traces"


def test_explicit_consent_survives_project_reload_and_child_environment(monkeypatch, tmp_path):
    from crewai_cli.experimental.eval_crew import turn_tracing_on
    from crewai_cli.run_crew import load_project_env
    from crewai_core.tool_credentials import build_env_with_all_tool_credentials

    monkeypatch.setattr(plus_api, "_MACHINE_AMP_ENV", {})
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "")
    monkeypatch.chdir(tmp_path)
    turn_tracing_on()
    load_project_env(tmp_path / ".env")
    assert os.environ["CREWAI_TRACING_ENABLED"] == "true"
    assert build_env_with_all_tool_credentials()["CREWAI_TRACING_ENABLED"] == "true"


def test_project_can_still_disable_machine_enabled_tracing(monkeypatch, tmp_path):
    from crewai_cli.run_crew import load_project_env

    monkeypatch.setattr(plus_api, "_MACHINE_AMP_ENV", {"CREWAI_TRACING_ENABLED": "true"})
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    env = tmp_path / ".env"
    env.write_text("CREWAI_TRACING_ENABLED=false\n")
    load_project_env(env)
    assert os.environ["CREWAI_TRACING_ENABLED"] == "false"
