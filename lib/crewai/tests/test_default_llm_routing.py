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
