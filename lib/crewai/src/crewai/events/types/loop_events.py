"""Events related to agent loop detection."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, model_validator

from crewai.events.base_events import BaseEvent


class LoopDetectedEvent(BaseEvent):
    """Event emitted when a repetitive loop pattern is detected in agent behavior.

    Attributes:
        agent_role: Role of the agent that is looping.
        repeated_tool: Description of the repeated tool call (name + args).
        action_taken: The action taken to break the loop.
        iteration: Current iteration count when the loop was detected.
    """

    agent_role: str
    agent_id: str | None = None
    task_id: str | None = None
    repeated_tool: str | None = None
    action_taken: str
    iteration: int
    agent: Any | None = None
    type: str = "loop_detected"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def set_fingerprint_data(self) -> LoopDetectedEvent:
        """Set fingerprint data from the agent if available."""
        if (
            self.agent
            and hasattr(self.agent, "fingerprint")
            and self.agent.fingerprint
        ):
            self.source_fingerprint = self.agent.fingerprint.uuid_str
            self.source_type = "agent"
            if (
                hasattr(self.agent.fingerprint, "metadata")
                and self.agent.fingerprint.metadata
            ):
                self.fingerprint_metadata = self.agent.fingerprint.metadata
        return self
