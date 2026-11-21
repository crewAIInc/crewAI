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
    # OpenAI-compatible providers crewAI runs natively; not offered by the
    # classic `crewai create` picker (PROVIDERS), but their keys are asked for
    # and checked like any other when a project's model uses them.
    "deepseek": [
        {
            "prompt": "Enter your DEEPSEEK API key (press Enter to skip)",
            "key_name": "DEEPSEEK_API_KEY",
        }
    ],
    "openrouter": [
        {
            "prompt": "Enter your OPENROUTER API key (press Enter to skip)",
            "key_name": "OPENROUTER_API_KEY",
        }
    ],
    "dashscope": [
        {
            "prompt": "Enter your DASHSCOPE API key (press Enter to skip)",
            "key_name": "DASHSCOPE_API_KEY",
        }
    ],
    "deepinfra": [
        {
            "prompt": "Enter your DEEPINFRA API key (press Enter to skip)",
            "key_name": "DEEPINFRA_API_KEY",
        }
    ],
    "wiro": [
        {
            "prompt": "Enter your WIRO API key (press Enter to skip)",
            "key_name": "WIRO_API_KEY",
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

# The crewai extra that holds a provider's SDK. `crewai create` adds it to the new
# project's crewai dependency; a provider not listed here (openai, ollama,
# cerebras) runs on what `crewai[tools]` installs.
PROVIDER_EXTRAS: dict[str, str] = {
    "anthropic": "anthropic",
    "gemini": "google-genai",
    "bedrock": "bedrock",
    "azure": "azure-ai-inference",
    # crewAI has no native provider for these and falls back to LiteLLM.
    "groq": "litellm",
    "huggingface": "litellm",
    "nvidia_nim": "litellm",
    "sambanova": "litellm",
    "watson": "litellm",
}

# The other prefixes crewAI routes to a native provider (PROVIDER_ALIASES in
# crewai/llm.py). The CLI does not depend on crewai, so it keeps this copy.
_PROVIDER_ALIASES: dict[str, str] = {
    "claude": "anthropic",
    "google": "gemini",
    "aws": "bedrock",
    "azure_openai": "azure",
    # Not a crewAI alias: models run as `watsonx/...`, and the CLI keys the
    # provider `watson`.
    "watsonx": "watson",
}


def provider_for_model(model: str) -> str | None:
    """Return the provider crewAI runs ``model`` on, or ``None`` if it is unknown.

    A ``provider/model`` string names its provider by prefix, with crewAI's
    aliases applied (``claude/`` is Anthropic). A bare name is recognized, in
    crewAI's order (``LLM._infer_provider_from_model``), by Bedrock's
    ``anthropic.`` namespace or an OpenAI, Anthropic or Gemini model prefix.
    """
    prefix, separator, _ = model.strip().lower().partition("/")
    if separator:
        return _PROVIDER_ALIASES.get(prefix, prefix) or None
    if "anthropic." in prefix:
        return "bedrock"
    if prefix.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if prefix.startswith("claude-"):
        return "anthropic"
    if prefix.startswith(_NATIVE_MODEL_PREFIXES["gemini"]):
        return "gemini"
    return None


def required_key_names(provider: str) -> list[str]:
    """Return the upper-cased ``ENV_VARS`` names ``provider`` needs set to run.

    Entries with a runtime default are left out, and so is azure's and
    cerebras's ``model`` entry, which fills ``MODEL`` and is not a credential.
    """
    names = [
        str(details["key_name"]).upper()
        for details in ENV_VARS.get(provider, [])
        if "key_name" in details and not details.get("optional", False)
    ]
    return [name for name in names if name != "MODEL"]


# The model names each native provider takes after its prefix, copied from
# LLM._matches_provider_pattern in crewai/llm.py. crewAI sends any other name
# under that prefix to LiteLLM. Bedrock takes any name with a ".".
_NATIVE_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "o4", "whisper-"),
    "anthropic": ("claude-", "anthropic."),
    "gemini": ("gemini-", "gemma-", "learnlm-"),
    "deepseek": ("deepseek",),
}

# The other native providers in crewAI's PROVIDER_ALIASES, which take every
# name (DeepInfra raises on one that is not org/model, so no extra helps there).
# crewAI sends a prefix it has no native provider for, such as `mistral/`, to
# LiteLLM.
_NATIVE_ANY_MODEL_PROVIDERS = frozenset(
    {
        "azure",
        "cerebras",
        "dashscope",
        "deepinfra",
        "hosted_vllm",
        "ollama",
        "ollama_chat",
        "openrouter",
        "snowflake",
        "wiro",
    }
)


def crewai_extras_for_model(model: str) -> set[str]:
    """Return the crewai extras, besides ``tools``, a project needs to run ``model``.

    That is the provider's extra from ``PROVIDER_EXTRAS``, plus ``litellm`` when
    crewAI sends a ``provider/model`` string to LiteLLM: because the native
    provider does not take that name, as with
    ``bedrock/claude-sonnet-4-5-20250929-v1:0`` or ``anthropic/my-finetune``, or
    because crewAI has no native provider for the prefix, as with
    ``mistral/mistral-large``.
    """
    provider = provider_for_model(model)
    extras: set[str] = set()
    if provider in PROVIDER_EXTRAS:
        extras.add(PROVIDER_EXTRAS[provider])
    _, separator, name = model.strip().lower().partition("/")
    if not separator:
        # crewAI runs a bare model name on the provider it infers.
        return extras
    if provider == "bedrock":
        native = "." in name
    elif provider in _NATIVE_MODEL_PREFIXES:
        native = name.startswith(_NATIVE_MODEL_PREFIXES[provider])
    else:
        native = provider in _NATIVE_ANY_MODEL_PROVIDERS
    if not native:
        extras.add("litellm")
    return extras


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

DEFAULT_LLM_MODEL = "gpt-6-luna"

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

LITELLM_PARAMS = ["api_key", "api_base", "api_version"]
