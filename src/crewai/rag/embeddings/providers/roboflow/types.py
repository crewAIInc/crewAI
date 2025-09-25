"""Type definitions for Roboflow embedding providers."""

from typing import Annotated, Literal, TypedDict


class RoboflowProviderConfig(TypedDict, total=False):
    """Configuration for Roboflow provider."""

    api_key: Annotated[str, ""]
    api_url: Annotated[str, "https://infer.roboflow.com"]


class RoboflowProviderSpec(TypedDict):
    """Roboflow provider specification."""

    provider: Literal["roboflow"]
    config: RoboflowProviderConfig
