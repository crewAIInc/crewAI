"""Tests for uploader factory."""

from crewai_files.processing.exceptions import PermanentUploadError
from crewai_files.uploaders.factory import get_uploader


class TestGetUploader:
    """Tests for the get_uploader factory function."""

    def test_bare_raise_bedrock_no_config(self):
        """get_uploader('bedrock') without bucket config raises PermanentUploadError,
        not RuntimeError."""
        raised = False
        try:
            get_uploader("bedrock")
        except PermanentUploadError:
            raised = True
        assert raised, "expected PermanentUploadError for unconfigured bedrock"

    def test_bare_raise_unknown_provider(self):
        """get_uploader('not-a-real-provider') raises PermanentUploadError,
        not RuntimeError."""
        raised = False
        try:
            get_uploader("not-a-real-provider")
        except PermanentUploadError:
            raised = True
        assert raised, "expected PermanentUploadError for unknown provider"

    def test_valid_providers_still_work(self):
        """Valid providers should still return a result or raise ImportError,
        not PermanentUploadError."""
        for provider in ("gemini", "openai", "anthropic"):
            try:
                result = get_uploader(provider)
                # If it returned, it's an uploader instance
                assert result is not None
            except ImportError:
                pass  # acceptable — the SDK isn't installed
            except PermanentUploadError:
                assert False, f"PermanentUploadError for valid provider {provider}"

    def test_unknown_provider_message(self):
        """Error message from unknown provider should include the provider name."""
        try:
            get_uploader("nonexistent-provider-xyz")
            assert False, "expected PermanentUploadError"
        except PermanentUploadError as e:
            assert "nonexistent-provider-xyz" in str(e)

    def test_bedrock_message(self):
        """Error message from unconfigured bedrock should mention the env var."""
        try:
            get_uploader("bedrock")
            assert False, "expected PermanentUploadError"
        except PermanentUploadError as e:
            assert "CREWAI_BEDROCK_S3_BUCKET" in str(e)
