import os
import subprocess
import sys
from unittest.mock import patch

from crewai.events.listeners.tracing.trace_batch_manager import TraceBatchManager


def test_constructing_batch_manager_does_not_touch_token_manager():
    with patch("crewai_core.auth.token.TokenManager") as token_manager:
        TraceBatchManager()

    token_manager.assert_not_called()


def test_plus_api_uses_saved_token_on_first_use():
    with patch("crewai_core.auth.token.TokenManager") as token_manager:
        token_manager.return_value.get_token.return_value = "saved-token"
        bm = TraceBatchManager()

        plus_api = bm.plus_api

    token_manager.assert_called_once()
    assert plus_api.api_key == "saved-token"
    assert bm.plus_api is plus_api


def test_plus_api_falls_back_to_anonymous_client_without_token():
    with patch("crewai_core.auth.token.TokenManager") as token_manager:
        token_manager.return_value.get_token.return_value = None
        plus_api = TraceBatchManager().plus_api

    assert plus_api.api_key is None


def test_plus_api_falls_back_to_anonymous_client_on_unreadable_token():
    with patch(
        "crewai_core.auth.token.TokenManager", side_effect=RuntimeError("bad key")
    ):
        plus_api = TraceBatchManager().plus_api

    assert plus_api.api_key is None


def test_importing_crewai_does_not_create_secret_key(tmp_path):
    code = (
        "from unittest.mock import patch\n"
        "with patch('crewai_core.token_manager.TokenManager._get_or_create_key',"
        " side_effect=AssertionError('key created at import')) as m:\n"
        "    import crewai.events.event_listener\n"
        "assert not m.called\n"
    )
    # Point every platform's data dir at tmp_path so the real credentials stay untouched.
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "LOCALAPPDATA": str(tmp_path / "localappdata"),
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert not list(tmp_path.rglob("secret.key"))
