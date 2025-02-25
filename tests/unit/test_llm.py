import pytest
from crewai.llm import LLM


@pytest.mark.parametrize(
    "invalid_model,error_message",
    [
        (3420, "Invalid model ID: 3420. Model ID cannot be a numeric value without a provider prefix."),
        ("3420", "Invalid model ID: 3420. Model ID cannot be a numeric value without a provider prefix."),
        (3.14, "Invalid model ID: 3.14. Model ID cannot be a numeric value without a provider prefix."),
    ],
)
def test_invalid_numeric_model_ids(invalid_model, error_message):
    """Test that numeric model IDs are rejected."""
    with pytest.raises(ValueError, match=error_message):
        LLM(model=invalid_model)


@pytest.mark.parametrize(
    "valid_model",
    [
        "openai/gpt-4",
        "gpt-3.5-turbo",
        "anthropic/claude-2",
    ],
)
def test_valid_model_ids(valid_model):
    """Test that valid model IDs are accepted."""
    llm = LLM(model=valid_model)
    assert llm.model == valid_model


def test_empty_model_id():
    """Test that empty model IDs are rejected."""
    with pytest.raises(ValueError, match="Invalid model ID: ''. Model ID cannot be empty or whitespace."):
        LLM(model="")


def test_whitespace_model_id():
    """Test that whitespace model IDs are rejected."""
    with pytest.raises(ValueError, match="Invalid model ID: '   '. Model ID cannot be empty or whitespace."):
        LLM(model="   ")
