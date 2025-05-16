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


PROVIDERS = [
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

MODELS = {
    "openai": [
        "gpt-4",
        "gpt-4.1",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano-2025-04-14",
        "gpt-4o",
        "gpt-4o-mini",
        "o1-mini",
        "o1-preview",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20240620",
        "claude-3-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ],
    "gemini": [
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-2.0-flash-lite-001",
        "gemini/gemini-2.0-flash-001",
        "gemini/gemini-2.0-flash-thinking-exp-01-21",
        "gemini/gemini-2.5-flash-preview-04-17",
        "gemini/gemini-2.5-pro-exp-03-25",
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

DEFAULT_LLM_MODEL = "gpt-4o-mini"

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


LITELLM_PARAMS = ["api_key", "api_base", "api_version"]
