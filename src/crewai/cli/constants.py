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

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"


LITELLM_PARAMS = ["api_key", "api_base", "api_version"]
