"""Tests for the a2a-sdk version compatibility shim (crewai.a2a._compat).

The shim must import cleanly regardless of whether a2a-sdk 0.3.x or 1.x is
installed, and each exported symbol must be usable.
"""

from __future__ import annotations

import pytest


class TestCompatImports:
    """Verify that all symbols exported by _compat are importable."""

    def test_type_symbols_importable(self) -> None:
        """All type symbols resolve without ImportError."""
        from crewai.a2a._compat import (
            DataPart,
            FilePart,
            FileWithBytes,
            FileWithUri,
            PushNotificationAuthenticationInfo,
            PushNotificationConfig,
            TaskIdParams,
            TaskQueryParams,
            TextPart,
        )

        assert DataPart is not None
        assert FilePart is not None
        assert FileWithBytes is not None
        assert FileWithUri is not None
        assert PushNotificationAuthenticationInfo is not None
        assert PushNotificationConfig is not None
        assert TaskIdParams is not None
        assert TaskQueryParams is not None
        assert TextPart is not None

    def test_error_symbol_importable(self) -> None:
        """A2AClientHTTPError resolves without ImportError."""
        from crewai.a2a._compat import A2AClientHTTPError

        assert A2AClientHTTPError is not None

    def test_server_error_importable(self) -> None:
        """ServerError resolves without ImportError."""
        from crewai.a2a._compat import ServerError

        assert ServerError is not None

    def test_utility_functions_importable(self) -> None:
        """All utility functions resolve without ImportError."""
        from crewai.a2a._compat import (
            get_data_parts,
            get_file_parts,
            new_agent_text_message,
            new_data_artifact,
            new_text_artifact,
        )

        assert callable(get_data_parts)
        assert callable(get_file_parts)
        assert callable(new_agent_text_message)
        assert callable(new_data_artifact)
        assert callable(new_text_artifact)

    def test_text_part_usable(self) -> None:
        """TextPart can be instantiated with a text field."""
        from crewai.a2a._compat import TextPart

        part = TextPart(text="hello")
        assert part.text == "hello"

    def test_server_error_usable(self) -> None:
        """ServerError can be raised and caught."""
        from crewai.a2a._compat import ServerError

        err = ServerError(error=None)
        assert str(err) == "None"

        with pytest.raises(ServerError):
            raise ServerError(error=None)
