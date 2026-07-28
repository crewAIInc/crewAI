"""Tests for the uploader factory.

Regression tests for #6568: bare ``raise`` statements in ``get_uploader()``
were outside any exception handler, causing ``RuntimeError: No active
exception to re-raise`` instead of a meaningful ``PermanentUploadError``.
"""

import builtins
import sys
from unittest.mock import patch

from crewai_files.processing.exceptions import PermanentUploadError
from crewai_files.uploaders.factory import get_uploader
import pytest


class TestUploaderFactory:
    """Tests for the get_uploader factory function."""

    def test_bedrock_without_config_raises_permanent_error(self):
        """get_uploader('bedrock') without bucket config raises
        PermanentUploadError, not RuntimeError."""
        with patch("os.environ.get", return_value=None):
            with pytest.raises(PermanentUploadError):
                get_uploader("bedrock")

    def test_unknown_provider_raises_permanent_error(self):
        """get_uploader('not-a-real-provider') raises PermanentUploadError,
        not RuntimeError."""
        with pytest.raises(PermanentUploadError):
            get_uploader("not-a-real-provider")

    def test_unknown_provider_message_includes_provider(self):
        """The error message should include the provider name."""
        with pytest.raises(PermanentUploadError) as exc_info:
            get_uploader("nonexistent-provider-xyz")
        assert "nonexistent-provider-xyz" in str(exc_info.value)

    def test_bedrock_message_mentions_env_var(self):
        """The error message should mention CREWAI_BEDROCK_S3_BUCKET."""
        with patch("os.environ.get", return_value=None):
            with pytest.raises(PermanentUploadError) as exc_info:
                get_uploader("bedrock")
        assert "CREWAI_BEDROCK_S3_BUCKET" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("provider", "sdk_module"),
        [
            ("gemini", "google.genai"),
            ("anthropic", "anthropic"),
            ("openai", "openai"),
            ("bedrock", "boto3"),
        ],
    )
    def test_missing_sdk_raises_permanent_error(self, provider, sdk_module):
        """An ImportError for a provider SDK should surface as
        PermanentUploadError, not a bare re-raise (RuntimeError)."""
        # Evict the provider SDK from sys.modules so the factory's lazy
        # ``from <sdk> import ...`` raises ImportError.
        top_level = sdk_module.split(".")[0]
        sdk_import = f"crewai_files.uploaders.{provider}"

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == top_level or name.startswith(f"{top_level}."):
                raise ImportError(f"No module named {name!r}")
            return real_import(name, *args, **kwargs)

        kwargs = {"bucket_name": "test-bucket"} if provider == "bedrock" else {}
        with patch("os.environ.get", return_value="test-bucket"), patch(
            "builtins.__import__", side_effect=fake_import
        ), patch.dict(sys.modules, {sdk_import: None}):
            with pytest.raises(PermanentUploadError):
                get_uploader(provider, **kwargs)
