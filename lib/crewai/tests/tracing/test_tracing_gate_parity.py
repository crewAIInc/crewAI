"""`crewai traces status` (and `crewai doctor`) report tracing through
`crewai_core.user_data.is_tracing_enabled`; it must agree with the gate a run
actually applies, `should_enable_tracing`."""

from pathlib import Path

from crewai.events.listeners.tracing.utils import should_enable_tracing
from crewai_core import user_data
import pytest


USER_DATA = {
    "nothing recorded": None,
    "consented": {"trace_consent": True, "first_execution_done": True},
    "declined": {"trace_consent": False, "first_execution_done": True},
}


@pytest.fixture
def isolated_user_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_STORAGE_DIR", "crewai_tracing_gate_parity")
    monkeypatch.setattr(
        "crewai_core.paths.appdirs.user_data_dir",
        lambda app, author: str(tmp_path / app),
    )
    monkeypatch.delenv("CREWAI_EVAL_AWAITING_RUN", raising=False)


@pytest.mark.usefixtures("isolated_user_data")
@pytest.mark.parametrize("env_value", [None, "true", "1", "TRUE", "false", "0"])
@pytest.mark.parametrize("recorded", list(USER_DATA))
def test_status_helper_matches_runtime_gate(
    monkeypatch: pytest.MonkeyPatch, env_value: str | None, recorded: str
) -> None:
    if env_value is None:
        monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CREWAI_TRACING_ENABLED", env_value)
    data = USER_DATA[recorded]
    if data is not None:
        user_data.update_user_data(data)

    assert user_data.is_tracing_enabled() is should_enable_tracing()


@pytest.mark.usefixtures("isolated_user_data")
def test_env_false_turns_tracing_off_despite_consent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_data.update_user_data(USER_DATA["consented"])
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")

    assert user_data.is_tracing_enabled() is False
    assert should_enable_tracing() is False


@pytest.mark.parametrize("pat", [None, "pat-token"])
@pytest.mark.parametrize("integration", [None, "integration-token"])
@pytest.mark.parametrize("logged_in", [True, False])
def test_doctor_names_the_credential_a_run_sends(
    monkeypatch: pytest.MonkeyPatch,
    pat: str | None,
    integration: str | None,
    logged_in: bool,
) -> None:
    """`crewai doctor` cannot import crewai, so it mirrors the order in which
    a run picks the credential its traces are sent with."""
    from crewai.telemetry.tracing import grants
    from crewai_cli.doctor import _tracing_credential_source

    for name, value in (
        ("CREWAI_USER_PAT", pat),
        ("CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN", integration),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    def get_auth_token() -> str:
        if logged_in:
            return "login-token"
        raise grants.AuthError("not logged in")

    monkeypatch.setattr(grants, "get_auth_token", get_auth_token)

    assert _tracing_credential_source(logged_in) == grants.tracing_credential_source()
