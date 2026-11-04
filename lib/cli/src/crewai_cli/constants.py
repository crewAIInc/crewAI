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
# under that prefix to LiteLLM. Bedrock takes any name with a "."; Azure, Ollama
# and Cerebras take every name.
_NATIVE_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "openai": ("gpt-", "o1", "o3", "o4", "whisper-"),
    "anthropic": ("claude-", "anthropic."),
    "gemini": ("gemini-", "gemma-", "learnlm-"),
    "deepseek": ("deepseek",),
}


def crewai_extras_for_model(model: str) -> set[str]:
    """Return the crewai extras, besides ``tools``, a project needs to run ``model``.

    That is the provider's extra from ``PROVIDER_EXTRAS``, plus ``litellm`` when
    crewAI sends a ``provider/model`` string to LiteLLM because the native
    provider does not take that name, as with
    ``bedrock/claude-sonnet-4-5-20250929-v1:0`` or ``anthropic/my-finetune``.
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
        native = True
    if not native:
        extras.add("litellm")
    return extras


MODELS: dict[str, list[str]] = {
    "openai": [
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-pro",
        "gpt-5.4-nano",
        "gpt-5.2",
        "gpt-5.2-pro",
        "gpt-5.1",
        "gpt-5",
        "gpt-5-pro",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "o4-mini",
        "o3",
        "o3-mini",
        "o1",
        "o1-mini",
        "o1-preview",
        "gpt-4",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
    ],
    "anthropic": [
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-haiku-4-5-20251001",
    ],
    "gemini": [
        "gemini/gemini-3.8-flash",
        "gemini/gemini-4-argon",
        "gemini/gemini-3.6-flash",
        "gemini/gemini-3.5-flash-lite",
        "gemini/gemini-gemma-2-9b-it",
        "gemini/gemini-gemma-2-27b-it",
        "gemini/gemma-3-1b-it",
        "gemini/gemma-3-4b-it",
        "gemini/gemma-3-12b-it",
        "gemini/gemma-3-27b-it",
    ],
    "nvidia_nim": [
        "nvidia_nim/nvidia/mistral-nemo-minitron-8b-8k-instruct",
        "nvidia_nim/nvidia/nemotron-4-mini-hindi-4b-instruct",
        "nvidia_nim/nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia_nim/nvidia/llama3-chatqa-1.5-8b",
        "nvidia_nim/nvidia/llama3-chatqa-1.5-70b",
        "nvidia_nim/nvidia/vila",
        "nvidia_nim/nvidia/neva-22",
        "nvidia_nim/nvidia/nemotron-mini-4b-instruct",
        "nvidia_nim/nvidia/usdcode-llama3-70b-instruct",
        "nvidia_nim/nvidia/nemotron-4-340b-instruct",
        "nvidia_nim/meta/codellama-70b",
        "nvidia_nim/meta/llama2-70b",
        "nvidia_nim/meta/llama3-8b-instruct",
        "nvidia_nim/meta/llama3-70b-instruct",
        "nvidia_nim/meta/llama-3.1-8b-instruct",
        "nvidia_nim/meta/llama-3.1-70b-instruct",
        "nvidia_nim/meta/llama-3.1-405b-instruct",
        "nvidia_nim/meta/llama-3.2-1b-instruct",
        "nvidia_nim/meta/llama-3.2-3b-instruct",
        "nvidia_nim/meta/llama-3.2-11b-vision-instruct",
        "nvidia_nim/meta/llama-3.2-90b-vision-instruct",
        "nvidia_nim/meta/llama-3.1-70b-instruct",
        "nvidia_nim/google/gemma-7b",
        "nvidia_nim/google/gemma-2b",
        "nvidia_nim/google/codegemma-7b",
        "nvidia_nim/google/codegemma-1.1-7b",
        "nvidia_nim/google/recurrentgemma-2b",
        "nvidia_nim/google/gemma-2-9b-it",
        "nvidia_nim/google/gemma-2-27b-it",
        "nvidia_nim/google/gemma-2-2b-it",
        "nvidia_nim/google/deplot",
        "nvidia_nim/google/paligemma",
        "nvidia_nim/mistralai/mistral-7b-instruct-v0.2",
        "nvidia_nim/mistralai/mixtral-8x7b-instruct-v0.1",
        "nvidia_nim/mistralai/mistral-large",
        "nvidia_nim/mistralai/mixtral-8x22b-instruct-v0.1",
        "nvidia_nim/mistralai/mistral-7b-instruct-v0.3",
        "nvidia_nim/nv-mistralai/mistral-nemo-12b-instruct",
        "nvidia_nim/mistralai/mamba-codestral-7b-v0.1",
        "nvidia_nim/microsoft/phi-3-mini-128k-instruct",
        "nvidia_nim/microsoft/phi-3-mini-4k-instruct",
        "nvidia_nim/microsoft/phi-3-small-8k-instruct",
        "nvidia_nim/microsoft/phi-3-small-128k-instruct",
        "nvidia_nim/microsoft/phi-3-medium-4k-instruct",
        "nvidia_nim/microsoft/phi-3-medium-128k-instruct",
        "nvidia_nim/microsoft/phi-3.5-mini-instruct",
        "nvidia_nim/microsoft/phi-3.5-moe-instruct",
        "nvidia_nim/microsoft/kosmos-2",
        "nvidia_nim/microsoft/phi-3-vision-128k-instruct",
        "nvidia_nim/microsoft/phi-3.5-vision-instruct",
        "nvidia_nim/databricks/dbrx-instruct",
        "nvidia_nim/snowflake/arctic",
        "nvidia_nim/aisingapore/sea-lion-7b-instruct",
        "nvidia_nim/ibm/granite-8b-code-instruct",
        "nvidia_nim/ibm/granite-34b-code-instruct",
        "nvidia_nim/ibm/granite-3.0-8b-instruct",
        "nvidia_nim/ibm/granite-3.0-3b-a800m-instruct",
        "nvidia_nim/mediatek/breeze-7b-instruct",
        "nvidia_nim/upstage/solar-10.7b-instruct",
        "nvidia_nim/writer/palmyra-med-70b-32k",
        "nvidia_nim/writer/palmyra-med-70b",
        "nvidia_nim/writer/palmyra-fin-70b-32k",
        "nvidia_nim/01-ai/yi-large",
        "nvidia_nim/deepseek-ai/deepseek-coder-6.7b-instruct",
        "nvidia_nim/rakuten/rakutenai-7b-instruct",
        "nvidia_nim/rakuten/rakutenai-7b-chat",
        "nvidia_nim/baichuan-inc/baichuan2-13b-chat",
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
        "watsonx/meta-llama/llama-3-1-70b-instruct",
        "watsonx/meta-llama/llama-3-1-8b-instruct",
        "watsonx/meta-llama/llama-3-2-11b-vision-instruct",
        "watsonx/meta-llama/llama-3-2-1b-instruct",
        "watsonx/meta-llama/llama-3-2-90b-vision-instruct",
        "watsonx/meta-llama/llama-3-405b-instruct",
        "watsonx/mistral/mistral-large",
        "watsonx/ibm/granite-3-8b-instruct",
    ],
    "bedrock": [
        "bedrock/us.amazon.nova-pro-v1:0",
        "bedrock/us.amazon.nova-micro-v1:0",
        "bedrock/us.amazon.nova-lite-v1:0",
        "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/us.anthropic.claude-3-opus-20240229-v1:0",
        "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/us.meta.llama3-2-11b-instruct-v1:0",
        "bedrock/us.meta.llama3-2-3b-instruct-v1:0",
        "bedrock/us.meta.llama3-2-90b-instruct-v1:0",
        "bedrock/us.meta.llama3-2-1b-instruct-v1:0",
        "bedrock/us.meta.llama3-1-8b-instruct-v1:0",
        "bedrock/us.meta.llama3-1-70b-instruct-v1:0",
        "bedrock/us.meta.llama3-3-70b-instruct-v1:0",
        "bedrock/us.meta.llama3-1-405b-instruct-v1:0",
        "bedrock/eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/eu.anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/eu.anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/eu.meta.llama3-2-3b-instruct-v1:0",
        "bedrock/eu.meta.llama3-2-1b-instruct-v1:0",
        "bedrock/apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/apac.anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/apac.anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/amazon.nova-pro-v1:0",
        "bedrock/amazon.nova-micro-v1:0",
        "bedrock/amazon.nova-lite-v1:0",
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
        "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/anthropic.claude-3-opus-20240229-v1:0",
        "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
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
        "bedrock/mistral.mistral-7b-instruct-v0:2",
        "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
    ],
    "huggingface": [
        "huggingface/meta-llama/Meta-Llama-3.1-8B-Instruct",
        "huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "huggingface/tiiuae/falcon-180B-chat",
        "huggingface/google/gemma-7b-it",
    ],
    "sambanova": [
        "sambanova/Meta-Llama-3.3-70B-Instruct",
        "sambanova/QwQ-32B-Preview",
        "sambanova/Qwen2.5-72B-Instruct",
        "sambanova/Qwen2.5-Coder-32B-Instruct",
        "sambanova/Meta-Llama-3.1-405B-Instruct",
        "sambanova/Meta-Llama-3.1-70B-Instruct",
        "sambanova/Meta-Llama-3.1-8B-Instruct",
        "sambanova/Llama-3.2-90B-Vision-Instruct",
        "sambanova/Llama-3.2-11B-Vision-Instruct",
        "sambanova/Meta-Llama-3.2-3B-Instruct",
        "sambanova/Meta-Llama-3.2-1B-Instruct",
    ],
}

DEFAULT_LLM_MODEL = "gpt-4.1-mini"

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"

LITELLM_PARAMS = ["api_key", "api_base", "api_version"]
