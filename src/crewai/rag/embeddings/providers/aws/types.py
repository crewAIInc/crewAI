"""Type definitions for AWS embedding providers."""

from typing import Annotated, Any, Literal, TypedDict


class BedrockProviderConfig(TypedDict, total=False):
    """Configuration for Bedrock provider."""

    model_name: Annotated[str, "amazon.titan-embed-text-v1"]
    session: Any


class BedrockProviderSpec(TypedDict):
    """Bedrock provider specification."""

    provider: Literal["amazon-bedrock"]
    config: BedrockProviderConfig
