"""Tests for provider data fetching."""

import json
import os

from crewai_cli import provider


class _FakeResponse:
    headers = {"content-length": "13"}

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self, chunk_size: int):
        yield b'{"openai": []}'


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response

    def __enter__(self) -> _FakeResponse:
        return self.response

    def __exit__(self, *args: object) -> bool:
        return False


def test_fetch_provider_data_keeps_configured_ca_bundle(tmp_path, monkeypatch):
    ca_bundle = str(tmp_path / "corp-ca.pem")
    monkeypatch.setenv("SSL_CERT_FILE", ca_bundle)
    seen = {}

    def fake_stream(method, url, timeout, verify):
        seen["verify"] = verify
        return _FakeStream(_FakeResponse())

    monkeypatch.setattr(provider.httpx, "stream", fake_stream)
    cache_file = tmp_path / "provider.json"

    assert provider.fetch_provider_data(cache_file) == {"openai": []}
    assert seen["verify"] == ca_bundle
    assert os.environ["SSL_CERT_FILE"] == ca_bundle
    assert json.loads(cache_file.read_text(encoding="utf-8")) == {"openai": []}
