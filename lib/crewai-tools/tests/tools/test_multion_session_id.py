"""MultiOnTool must keep a session id supplied by the caller."""

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("multion", MagicMock())
sys.modules.setdefault("multion.client", MagicMock())

from crewai_tools import MultiOnTool  # noqa: E402


def test_session_id_is_preserved():
    tool = MultiOnTool(api_key="key", session_id="existing-session")

    assert tool.session_id == "existing-session"
