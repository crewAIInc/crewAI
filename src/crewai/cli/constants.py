from typing import Dict, List, Optional

ENV_VARS = {
    "openai": [
        {
            "prompt": "Enter your OPENAI API key (press Enter to skip)",
            "key_name": "OPENAI_API_KEY",
        }
    ],
    "anthropic": [
        {
            "prompt": "Enter your ANTHROPIC API key (press Enter to skip)",
            "key_name": "ANTHROPIC_API_KEY",
        }
    ],
    "gemini": [
        {
            "prompt": "Enter your GEMINI API key (press Enter to skip)",
            "key_name": "GEMINI_API_KEY",
        }
    ],
    "groq": [
        {
            "prompt": "Enter your GROQ API key (press Enter to skip)",
            "key_name": "GROQ_API_KEY",
        }
    ],
    "watson": [
        {
            "prompt": "Enter your WATSONX URL (press Enter to skip)",
            "key_name": "WATSONX_URL",
        },
        {
            "prompt": "Enter your WATSONX API Key (press Enter to skip)",
            "key_name": "WATSONX_APIKEY",
        },
        {
            "prompt": "Enter your WATSONX Project Id (press Enter to skip)",
            "key_name": "WATSONX_PROJECT_ID",
        },
    ],
    "ollama": [
        {
            "default": True,
            "API_BASE": "http://localhost:11434",
        }
    ],
    "bedrock": [
        {
            "prompt": "Enter your AWS Access Key ID (press Enter to skip)",
            "key_name": "AWS_ACCESS_KEY_ID",
        },
        {
            "prompt": "Enter your AWS Secret Access Key (press Enter to skip)",
            "key_name": "AWS_SECRET_ACCESS_KEY",
        },
        {
            "prompt": "Enter your AWS Region Name (press Enter to skip)",
            "key_name": "AWS_REGION_NAME",
        },
    ],
    "azure": [
        {
            "prompt": "Enter your Azure deployment name (must start with 'azure/')",
            "key_name": "model",
        },
        {
            "prompt": "Enter your AZURE API key (press Enter to skip)",
            "key_name": "AZURE_API_KEY",
        },
        {
            "prompt": "Enter your AZURE API base URL (press Enter to skip)",
            "key_name": "AZURE_API_BASE",
        },
        {
            "prompt": "Enter your AZURE API version (press Enter to skip)",
            "key_name": "AZURE_API_VERSION",
        },
    ],
    "cerebras": [
        {
            "prompt": "Enter your Cerebras model name (must start with 'cerebras/')",
            "key_name": "model",
        },
        {
            "prompt": "Enter your Cerebras API version (press Enter to skip)",
            "key_name": "CEREBRAS_API_KEY",
        },
    ],
}


PROVIDERS = [
    "openai",
    "anthropic",
    "gemini",
    "groq",
    "ollama",
    "watson",
    "bedrock",
    "azure",
    "cerebras",
]


MODELS = {
    "openai": ["gpt-4", "gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"],
    "anthropic": [
        "claude-3-5-sonnet-20240620",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ],
    "gemini": [
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-gemma-2-9b-it",
        "gemini/gemini-gemma-2-27b-it",
    ],
    "groq": [
        "groq/llama-3.1-8b-instant",
        "groq/llama-3.1-70b-versatile",
        "groq/llama-3.1-405b-reasoning",
        "groq/gemma2-9b-it",
        "groq/gemma-7b-it",
    ],
    "ollama": ["ollama/llama3.1", "ollama/mixtral"],
    "watson": [
        "watsonx/google/flan-t5-xxl",
        "watsonx/google/flan-ul2",
        "watsonx/bigscience/mt0-xxl",
        "watsonx/eleutherai/gpt-neox-20b",
        "watsonx/ibm/mpt-7b-instruct2",
        "watsonx/bigcode/starcoder",
        "watsonx/meta-llama/llama-2-70b-chat",
        "watsonx/meta-llama/llama-2-13b-chat",
        "watsonx/ibm/granite-13b-instruct-v1",
        "watsonx/ibm/granite-13b-chat-v1",
        "watsonx/google/flan-t5-xl",
        "watsonx/ibm/granite-13b-chat-v2",
        "watsonx/ibm/granite-13b-instruct-v2",
        "watsonx/elyza/elyza-japanese-llama-2-7b-instruct",
        "watsonx/ibm-mistralai/mixtral-8x7b-instruct-v01-q",
    ],
    "bedrock": [
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/anthropic.claude-3-opus-20240229-v1:0",
        "bedrock/anthropic.claude-v2:1",
        "bedrock/anthropic.claude-v2",
        "bedrock/anthropic.claude-instant-v1",
        "bedrock/meta.llama3-1-405b-instruct-v1:0",
        "bedrock/meta.llama3-1-70b-instruct-v1:0",
        "bedrock/meta.llama3-1-8b-instruct-v1:0",
        "bedrock/meta.llama3-70b-instruct-v1:0",
        "bedrock/meta.llama3-8b-instruct-v1:0",
        "bedrock/amazon.titan-text-lite-v1",
        "bedrock/amazon.titan-text-express-v1",
        "bedrock/cohere.command-text-v14",
        "bedrock/ai21.j2-mid-v1",
        "bedrock/ai21.j2-ultra-v1",
        "bedrock/ai21.jamba-instruct-v1:0",
        "bedrock/meta.llama2-13b-chat-v1",
        "bedrock/meta.llama2-70b-chat-v1",
        "bedrock/mistral.mistral-7b-instruct-v0:2",
        "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
    ],
}


def get_env_vars(provider: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get environment variables configuration for specified provider or all providers.
    
    Args:
        provider (str, optional): The provider name to get env vars for. Defaults to None.
    
    Returns:
        dict: Environment variables configuration
    """
    if provider is not None and not isinstance(provider,str):
        raise TypeError("Provider must be a string or None")
    if provider and provider not in ENV_VARS:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ENV_VARS[provider] if provider else ENV_VARS


def get_providers() -> list:
    """
    Get list of all supported providers.
    
    Returns:
        list: List of provider names
    """
    return PROVIDERS


def get_models(provider: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Get available models for specified provider or all providers.
    
    Args:
        provider (str, optional): The provider name to get models for. Defaults to None.
    
    Returns:
        dict: Available models configuration
    """
    if provider and provider in MODELS:
        return {provider: MODELS[provider]}
    return MODELS


JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

# Tests
def test_get_env_vars():
    assert get_env_vars() is not None
    assert isinstance(get_env_vars("openai"), list)
    with pytest.raises(ValueError):
        get_env_vars("invalid_provider")
