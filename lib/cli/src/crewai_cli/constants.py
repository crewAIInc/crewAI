from typing import Any


DEFAULT_CREWAI_ENTERPRISE_URL = "https://app.crewai.com"
CREWAI_ENTERPRISE_DEFAULT_OAUTH2_PROVIDER = "workos"
CREWAI_ENTERPRISE_DEFAULT_OAUTH2_AUDIENCE = "client_01JNJQWBJ4SPFN3SWJM5T7BDG8"
CREWAI_ENTERPRISE_DEFAULT_OAUTH2_CLIENT_ID = "client_01JYT06R59SP0NXYGD994NFXXX"
CREWAI_ENTERPRISE_DEFAULT_OAUTH2_DOMAIN = "login.crewai.com"

# `crewai create` hides what is typed at a prompt unless its entry sets
# "secret": False. An entry with "optional": True has a runtime default, so
# leaving it blank is not reported as a missing key.
ENV_VARS: dict[str, list[dict[str, Any]]] = {
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
            "prompt": "Enter your GEMINI API key from https://ai.dev/apikey (press Enter to skip)",
            "key_name": "GEMINI_API_KEY",
        }
    ],
    "nvidia_nim": [
        {
            "prompt": "Enter your NVIDIA API key (press Enter to skip)",
            "key_name": "NVIDIA_NIM_API_KEY",
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
            "secret": False,
        },
        {
            "prompt": "Enter your WATSONX API Key (press Enter to skip)",
            "key_name": "WATSONX_APIKEY",
        },
        {
            "prompt": "Enter your WATSONX Project Id (press Enter to skip)",
            "key_name": "WATSONX_PROJECT_ID",
            "secret": False,
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
            "key_name": "AWS_DEFAULT_REGION",
            "secret": False,
            "optional": True,  # BedrockCompletion falls back to us-east-1.
        },
    ],
    "azure": [
        {
            "prompt": "Enter your Azure deployment name (must start with 'azure/')",
            "key_name": "model",
            "secret": False,
        },
        {
            "prompt": "Enter your AZURE API key (press Enter to skip)",
            "key_name": "AZURE_API_KEY",
        },
        {
            "prompt": "Enter your AZURE API base URL (press Enter to skip)",
            "key_name": "AZURE_API_BASE",
            "secret": False,
        },
        {
            "prompt": "Enter your AZURE API version (press Enter to skip)",
            "key_name": "AZURE_API_VERSION",
            "secret": False,
            "optional": True,  # AzureCompletion falls back to 2024-06-01.
        },
    ],
    "cerebras": [
        {
            "prompt": "Enter your Cerebras model name (must start with 'cerebras/')",
            "key_name": "model",
            "secret": False,
        },
        {
            "prompt": "Enter your Cerebras API version (press Enter to skip)",
            "key_name": "CEREBRAS_API_KEY",
        },
    ],
    "huggingface": [
        {
            "prompt": "Enter your Huggingface API key (HF_TOKEN) (press Enter to skip)",
            "key_name": "HF_TOKEN",
        },
    ],
    "sambanova": [
        {
            "prompt": "Enter your SambaNovaCloud API key (press Enter to skip)",
            "key_name": "SAMBANOVA_API_KEY",
        }
    ],
}


PROVIDERS: list[str] = [
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

MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-6-luna",
        "gpt-6.1-sol",
        "gpt-6-astra",
    ],
    "anthropic": [
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-opus-5-5",
        "claude-fable-5-1",
    ],
    "gemini": [
        "gemini/gemini-3.8-flash",
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-4-argon",
    ],
    "nvidia_nim": [
        "nvidia_nim/nvidia/nemotron-3.5-super-vl-120b-a12b",
        "nvidia_nim/nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia_nim/nvidia/nemotron-3.5-lightning-30b-a3b",
    ],
    "groq": [
        "groq/openai/gpt-oss-20b",
        "groq/openai/gpt-oss-120b",
        "groq/qwen/qwen3.8-27b",
    ],
    "ollama": [
        "ollama/qwen3.5:9b",
        "ollama/gemma4:26b",
        "ollama/qwen3.8:27b",
        "ollama/gpt-oss:120b",
    ],
    "watson": [
        "watsonx/ibm/granite-4-h-small",
        "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8",
        "watsonx/openai/gpt-oss-120b",
    ],
    "bedrock": [
        "bedrock/global.anthropic.claude-sonnet-5-5",
        "bedrock/global.anthropic.claude-haiku-5-5",
        "bedrock/global.anthropic.claude-opus-5-5",
    ],
    "huggingface": [
        "huggingface/zai-org/GLM-5.3-Flash",
        "huggingface/deepseek-ai/DeepSeek-V4.1-Flash",
        "huggingface/zai-org/GLM-5.3",
    ],
    "sambanova": [
        "sambanova/gpt-oss-120b",
        "sambanova/DeepSeek-V3.1",
        "sambanova/gemma-4-31B-it",
    ],
}

DEFAULT_LLM_MODEL = "gpt-4.1-mini"

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

LITELLM_PARAMS = ["api_key", "api_base", "api_version"]
