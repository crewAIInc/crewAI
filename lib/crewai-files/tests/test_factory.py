"""Tests for get_uploader."""

from crewai_files.uploaders import get_uploader


def test_get_uploader_returns_none_for_unknown_provider():
    # Regression: an unsupported provider must return None (the callers all
    # branch on `if uploader is None`), not raise "RuntimeError: No active
    # exception to reraise" from a bare `raise`
    assert get_uploader("does-not-exist") is None


def test_get_uploader_returns_none_for_unconfigured_bedrock(monkeypatch):
    # Regression: Bedrock without a configured S3 bucket must return None,
    # not raise "RuntimeError: No active exception to reraise"
    monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)
    assert get_uploader("bedrock") is None


def test_get_uploader_returns_none_for_bedrock_with_falsy_bucket_name(monkeypatch):
    # An explicit falsy bucket_name (None or "") is unconfigured just like an
    # absent one, so the config guard must key on the value, not key presence
    monkeypatch.delenv("CREWAI_BEDROCK_S3_BUCKET", raising=False)
    assert get_uploader("bedrock", bucket_name=None) is None
    assert get_uploader("bedrock", bucket_name="") is None
