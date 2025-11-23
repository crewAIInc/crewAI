from crewai.llm.constants import SupportedNativeProviders


PROVIDER_MAPPING: dict[str, SupportedNativeProviders] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
    "azure": "azure",
    "azure_openai": "azure",
    "google": "gemini",
    "gemini": "gemini",
    "bedrock": "bedrock",
    "aws": "bedrock",
}
