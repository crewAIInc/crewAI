import pytest

pytest.importorskip("anthropic")

from crewai.llm import LLM

FOUNDRY = "https://res.services.ai.azure.com"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("AZURE_ENDPOINT", "AZURE_API_BASE"):
        monkeypatch.delenv(name, raising=False)


def _base_url(**kwargs: str) -> str:
    llm = LLM(model="azure/claude-sonnet-4-6", api_key="test-key", **kwargs)
    return llm._get_client_params()["base_url"]


@pytest.mark.parametrize(
    "endpoint",
    [FOUNDRY, f"{FOUNDRY}/", f"{FOUNDRY}/models", f"{FOUNDRY}/anthropic/"],
)
def test_endpoint_path_is_replaced_with_anthropic(endpoint: str) -> None:
    assert _base_url(endpoint=endpoint) == f"{FOUNDRY}/anthropic"


def test_endpoint_is_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_ENDPOINT", FOUNDRY)

    assert _base_url() == f"{FOUNDRY}/anthropic"


def test_endpoint_argument_wins_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_ENDPOINT", "https://other.services.ai.azure.com")

    assert _base_url(endpoint=FOUNDRY) == f"{FOUNDRY}/anthropic"


def test_explicit_base_url_is_used_as_given() -> None:
    private = "https://claude.internal.example.com/anthropic"

    assert _base_url(base_url=private, endpoint=FOUNDRY) == private


def test_openai_host_is_rejected_when_the_client_is_built() -> None:
    llm = LLM(
        model="azure/claude-sonnet-4-6",
        api_key="test-key",
        endpoint="https://res.openai.azure.com/openai/deployments/claude",
    )

    with pytest.raises(ValueError, match="Pass base_url explicitly"):
        llm._get_client_params()


def test_missing_endpoint_is_reported_on_first_use_not_construction() -> None:
    llm = LLM(model="azure/claude-sonnet-4-6", api_key="test-key")

    with pytest.raises(ValueError, match="Azure endpoint is required"):
        llm._get_client_params()
