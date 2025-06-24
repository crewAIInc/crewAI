import re
from pathlib import Path

from crewai.cli.constants import PROVIDERS


def test_cli_documentation_matches_providers():
    """Test that CLI documentation accurately reflects the available providers."""
    docs_path = Path(__file__).parent.parent / "docs" / "concepts" / "cli.mdx"
    with open(docs_path, 'r') as f:
        docs_content = f.read()
    
    assert "top 5" not in docs_content.lower(), "Documentation should not mention 'top 5' providers"
    assert "5 most common" not in docs_content.lower(), "Documentation should not mention '5 most common' providers"
    
    assert "list of available LLM providers" in docs_content or "following LLM providers" in docs_content, \
        "Documentation should mention the availability of multiple LLM providers"
    
    assert len(PROVIDERS) > 5, f"Expected more than 5 providers, but found {len(PROVIDERS)}"
    
    key_providers = ["OpenAI", "Anthropic", "Gemini"]
    for provider in key_providers:
        assert provider in docs_content, f"Key provider {provider} should be mentioned in documentation"


def test_providers_list_matches_constants():
    """Test that the actual PROVIDERS list has the expected providers."""
    expected_providers = [
        "openai",
        "anthropic", 
        "gemini",
        "nvidia_nim",
        "groq",
        "huggingface",
        "ollama",
        "watson",
        "bedrock",
        "azure",
        "cerebras",
        "sambanova",
    ]
    
    assert PROVIDERS == expected_providers, f"PROVIDERS list has changed. Expected {expected_providers}, got {PROVIDERS}"
    assert len(PROVIDERS) == 12, f"Expected 12 providers, but found {len(PROVIDERS)}"
