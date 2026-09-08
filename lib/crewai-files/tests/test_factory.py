"""Tests for get_uploader."""

from crewai_files.uploaders import get_uploader
import pytest


def test_get_uploader_raises_for_unknown_provider():
    # Regression for #7282: an unsupported provider must raise a clear
    # ValueError, not the opaque "RuntimeError: No active exception to reraise"
    # a bare `raise` produced, and not a silent None that hides the
    # misconfiguration behind an inline fallback
    with pytest.raises(ValueError, match="No file uploader available"):
        get_uploader("does-not-exist")


def test_get_uploader_raises_for_unconfigured_bedrock(monkeypatch):
    # Bedrock without a configured S3 bucket must raise a ValueError that names
    # the missing configuration, not RuntimeError and not a silent None
    monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)
    with pytest.raises(ValueError, match="CREWAI_BEDROCK_S3_BUCKET"):
        get_uploader("bedrock")


def test_get_uploader_raises_for_bedrock_with_falsy_bucket_name(monkeypatch):
    # An explicit falsy bucket_name (None or "") is unconfigured just like an
    # absent one, so the guard keys on the value, not key presence
    monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)
    with pytest.raises(ValueError, match="CREWAI_BEDROCK_S3_BUCKET"):
        get_uploader("bedrock", bucket_name=None)
    with pytest.raises(ValueError, match="CREWAI_BEDROCK_S3_BUCKET"):
        get_uploader("bedrock", bucket_name="")
