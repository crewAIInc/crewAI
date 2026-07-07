"""Tests for platform-API TLS verification behavior."""

import warnings

import pytest

from crewai_tools.tools.crewai_platform_tools.misc import platform_tls_verify


@pytest.fixture(autouse=True)
def _clear_tls_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CREWAI_FACTORY", raising=False)
    monkeypatch.delenv("CREWAI_PLATFORM_INSECURE_SKIP_TLS_VERIFY", raising=False)


def test_tls_verify_on_by_default() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would fail the test
        assert platform_tls_verify() is True


def test_tls_verify_disabled_by_explicit_flag_warns_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CREWAI_PLATFORM_INSECURE_SKIP_TLS_VERIFY", "true")
    with pytest.warns(UserWarning, match="TLS certificate verification is DISABLED"):
        assert platform_tls_verify() is False


def test_tls_verify_disabled_by_legacy_factory_flag_warns_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Back-compat: CREWAI_FACTORY still disables, but now loudly instead of silently.
    monkeypatch.setenv("CREWAI_FACTORY", "true")
    with pytest.warns(UserWarning, match="TLS certificate verification is DISABLED"):
        assert platform_tls_verify() is False


def test_tls_verify_only_exact_true_disables(monkeypatch: pytest.MonkeyPatch) -> None:
    # A stray non-"true" value must not silently drop TLS verification.
    monkeypatch.setenv("CREWAI_FACTORY", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert platform_tls_verify() is True
