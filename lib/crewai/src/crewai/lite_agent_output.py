"""Output class for LiteAgent execution results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from crewai.utilities.types import LLMMessage


class LiteAgentOutput(BaseModel):
    """Class that represents the result of a LiteAgent execution."""

    model_config = {"arbitrary_types_allowed": True}

    raw: str = Field(description="Raw output of the agent", default="")
    pydantic: BaseModel | None = Field(
        description="Pydantic output of the agent", default=None
    )
    agent_role: str = Field(description="Role of the agent that produced this output")
    usage_metrics: dict[str, Any] | None = Field(
        description="Token usage metrics for this execution", default=None
    )
    messages: list[LLMMessage] = Field(description="Messages of the agent", default=[])

    def to_dict(self) -> dict[str, Any]:
        """Convert pydantic_output to a dictionary."""
        if self.pydantic:
            return self.pydantic.model_dump()
        return {}

    def __str__(self) -> str:
        """Return the raw output as a string."""
        return self.raw
