import json
from pathlib import Path

from crewai_core import user_data
import pytest


@pytest.fixture
def user_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the production user-data helpers from real user configuration."""
    monkeypatch.setattr(user_data, "db_storage_path", lambda: str(tmp_path))
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    return tmp_path / ".crewai_user.json"


@pytest.mark.parametrize("raw", ["[]", "null", "false", "42", '"unexpected"'])
def test_non_object_user_data_does_not_break_status(user_file: Path, raw: str) -> None:
    """Treat structurally invalid JSON as corrupt without rewriting on read."""
    user_file.write_text(raw)

    assert user_data.has_user_declined_tracing() is False
    assert user_data.is_tracing_enabled() is False
    assert user_file.read_text() == raw


@pytest.mark.parametrize("raw", ["[]", "null", "false", "42", '"unexpected"'])
def test_update_recovers_from_non_object_user_data(user_file: Path, raw: str) -> None:
    """Allow a user to record tracing preferences despite a corrupt root value."""
    user_file.write_text(raw)
    updates = {"first_execution_done": True, "trace_consent": False}

    user_data.update_user_data(updates)

    assert json.loads(user_file.read_text()) == updates
    assert user_data.has_user_declined_tracing() is True
    assert user_data.is_tracing_enabled() is False


def test_update_preserves_existing_user_fields(user_file: Path) -> None:
    """Continue merging updates into valid object-shaped configuration."""
    user_file.write_text(json.dumps({"custom": "preserved", "trace_consent": True}))

    user_data.update_user_data({"first_execution_done": True, "trace_consent": False})

    assert json.loads(user_file.read_text()) == {
        "custom": "preserved",
        "first_execution_done": True,
        "trace_consent": False,
    }
