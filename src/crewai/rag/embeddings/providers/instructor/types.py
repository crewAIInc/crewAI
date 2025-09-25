"""Type definitions for Instructor embedding providers."""

from typing import Annotated, Literal, TypedDict


class InstructorProviderConfig(TypedDict, total=False):
    """Configuration for Instructor provider."""

    model_name: Annotated[str, "hkunlp/instructor-base"]
    device: Annotated[str, "cpu"]
    instruction: str


class InstructorProviderSpec(TypedDict):
    """Instructor provider specification."""

    provider: Literal["instructor"]
    config: InstructorProviderConfig
