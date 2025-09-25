"""Type definitions for AWS embedding providers."""

from typing import Annotated, Any, Literal, TypedDict

from typing_extensions import Required


class BedrockProviderConfig(TypedDict, total=False):
    """Configuration for Bedrock provider."""

    model_name: Annotated[str, "amazon.titan-embed-text-v1"]
    session: Any


class BedrockProviderSpec(TypedDict, total=False):
    """Bedrock provider specification."""

    provider: Required[Literal["amazon-bedrock"]]
    config: BedrockProviderConfig
