import pytest

pytest.importorskip("anthropic")

from crewai.llm import LLM

ENDPOINT = "https://res.services.ai.azure.com"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_API_BASE", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _llm(**kwargs: str) -> LLM:
    return LLM(model="azure/claude-sonnet-4-6", endpoint=ENDPOINT, **kwargs)


def test_key_is_read_from_azure_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")

    assert _llm()._get_client_params()["api_key"] == "azure-key"


def test_api_key_argument_wins_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")

    assert _llm(api_key="explicit")._get_client_params()["api_key"] == "explicit"


def test_anthropic_api_key_is_never_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-anthropic-key")

    with pytest.raises(ValueError, match="AZURE_API_KEY is required"):
        _llm()._get_client_params()


def test_azure_key_wins_when_both_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-anthropic-key")
    monkeypatch.setenv("AZURE_API_KEY", "azure-key")

    assert _llm()._get_client_params()["api_key"] == "azure-key"


def test_missing_key_is_reported_on_first_use_not_construction() -> None:
    llm = _llm()

    with pytest.raises(ValueError, match="AZURE_API_KEY is required"):
        llm._get_client_params()


def test_key_set_after_construction_is_picked_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm = _llm()
    monkeypatch.setenv("AZURE_API_KEY", "late-key")

    assert llm._get_client_params()["api_key"] == "late-key"
