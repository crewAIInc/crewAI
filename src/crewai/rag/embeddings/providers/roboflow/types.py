"""Type definitions for Roboflow embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class RoboflowProviderConfig(TypedDict, total=False):
    """Configuration for Roboflow provider."""

    api_key: Annotated[str, ""]
    api_url: Annotated[str, "https://infer.roboflow.com"]


class RoboflowProviderSpec(TypedDict):
    """Roboflow provider specification."""

    provider: Required[Literal["roboflow"]]
    config: RoboflowProviderConfig
