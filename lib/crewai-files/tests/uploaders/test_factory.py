"""Tests for get_uploader error handling."""

import pytest

from crewai_files.processing.exceptions import PermanentUploadError
from crewai_files.uploaders.factory import get_uploader


def test_get_uploader_unsupported_provider_raises_permanent_upload_error() -> None:
    """
    An unsupported provider must raise PermanentUploadError, not a bare
    ``RuntimeError: No active exception to re-raise``.

    The final ``raise`` in get_uploader was a bare re-raise outside any
    except block, so Python raised RuntimeError instead of a meaningful error.
    """
    with pytest.raises(PermanentUploadError, match="not-a-real-provider"):
        get_uploader("not-a-real-provider")


def test_get_uploader_bedrock_without_bucket_raises_permanent_upload_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Requesting the Bedrock uploader without a configured bucket must raise
    PermanentUploadError (a config error), not a bare RuntimeError.
    """
    monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)
    with pytest.raises(PermanentUploadError, match="CREWAI_BEDROCK_S3_BUCKET"):
        get_uploader("bedrock")
