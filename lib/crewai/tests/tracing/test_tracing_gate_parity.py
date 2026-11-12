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
