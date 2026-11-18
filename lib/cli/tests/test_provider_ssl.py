"""Tests for the CLI provider fetch and its SSL CA bundle handling."""

from __future__ import annotations

import os
from pathlib import Path

import certifi
import httpx
import pytest

import crewai_cli.provider as provider


class _FailingStream:
    """A stream context that fails before any request reaches the network."""

    def __enter__(self):
        raise httpx.HTTPError("stubbed network failure")

    def __exit__(self, *args):
        return False


def _stub_stream(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    def fake_stream(method: str, url: str, **kwargs) -> _FailingStream:
        captured["verify"] = kwargs.get("verify")
        return _FailingStream()

    monkeypatch.setattr(provider.httpx, "stream", fake_stream)
    monkeypatch.setattr(provider.click, "secho", lambda *args, **kwargs: None)


def test_custom_ca_bundle_is_used_and_left_in_place(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ca = tmp_path / "corp-ca.pem"
    ca.write_text("stub CA bundle", encoding="utf-8")
    monkeypatch.setenv("SSL_CERT_FILE", str(ca))
    captured: dict = {}
    _stub_stream(monkeypatch, captured)

    assert provider.fetch_provider_data(tmp_path / "providers.json") is None

    assert os.environ["SSL_CERT_FILE"] == str(ca)
    assert captured["verify"] == str(ca)


def test_certifi_is_used_when_no_ca_bundle_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    captured: dict = {}
    _stub_stream(monkeypatch, captured)

    assert provider.fetch_provider_data(tmp_path / "providers.json") is None

    assert "SSL_CERT_FILE" not in os.environ
    assert captured["verify"] == certifi.where()
