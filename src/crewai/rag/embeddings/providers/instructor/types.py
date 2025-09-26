"""Type definitions for Instructor embedding providers."""

from typing import Annotated, Literal

from typing_extensions import Required, TypedDict


class InstructorProviderConfig(TypedDict, total=False):
    """Configuration for Instructor provider."""

    model_name: Annotated[str, "hkunlp/instructor-base"]
    device: Annotated[str, "cpu"]
    instruction: str


class InstructorProviderSpec(TypedDict, total=False):
    """Instructor provider specification."""

    provider: Required[Literal["instructor"]]
    config: InstructorProviderConfig
