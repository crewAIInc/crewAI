from crewai.llm import LLM
import pytest


@pytest.mark.parametrize("model", ["gpt-5.6-luna", "openai/gpt-5.6-luna"])
def test_luna_default_models_use_native_openai_without_litellm(model, monkeypatch):
    """Runtime and CLI defaults work through native OpenAI without LiteLLM."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr("crewai.llm.LITELLM_AVAILABLE", False)

    llm = LLM(model=model)

    assert llm.is_litellm is False
    assert llm.provider == "openai"
    assert llm.model == "gpt-5.6-luna"


@pytest.mark.parametrize("model", [None, "gpt-4o"])
def test_openai_completion_default_preserves_explicit_model(model, monkeypatch):
    from crewai.llms.providers.openai.completion import OpenAICompletion

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    llm = OpenAICompletion(**({"model": model} if model else {}))

    assert llm.model == (model or "gpt-5.6-luna")
    assert llm.is_gpt4_model is (model == "gpt-4o")
