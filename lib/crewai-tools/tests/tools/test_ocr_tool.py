import pytest

from crewai_tools import OCRTool


def test_without_an_llm_it_reads_with_gpt_6_luna_at_the_default_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPT-6 Luna rejects any temperature but the default with a 400."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    llm = OCRTool().llm

    assert llm.model == "gpt-6-luna"
    assert llm.temperature is None
