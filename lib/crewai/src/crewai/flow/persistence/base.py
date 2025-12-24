"""Base class for flow state persistence."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from crewai.flow.async_feedback.types import PendingFeedbackContext


class FlowPersistence(ABC):
    """Abstract base class for flow state persistence.

    This class defines the interface that all persistence implementations must follow.
    It supports both structured (Pydantic BaseModel) and unstructured (dict) states.

    For async human feedback support, implementations can optionally override:
    - save_pending_feedback(): Saves state with pending feedback context
    - load_pending_feedback(): Loads state and pending feedback context
    - clear_pending_feedback(): Clears pending feedback after resume
    """

    @abstractmethod
    def init_db(self) -> None:
        """Initialize the persistence backend.

        This method should handle any necessary setup, such as:
        - Creating tables
        - Establishing connections
        - Setting up indexes
        """

    @abstractmethod
    def save_state(
        self, flow_uuid: str, method_name: str, state_data: dict[str, Any] | BaseModel
    ) -> None:
        """Persist the flow state after method completion.

        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
            state_data: Current state data (either dict or Pydantic model)
        """

    @abstractmethod
    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        """Load the most recent state for a given flow UUID.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            The most recent state as a dictionary, or None if no state exists
        """

    def save_pending_feedback(
        self,
        flow_uuid: str,
        context: PendingFeedbackContext,
        state_data: dict[str, Any] | BaseModel,
    ) -> None:
        """Save state with a pending feedback marker.

        This method is called when a flow is paused waiting for async human
        feedback. The default implementation just saves the state without
        the pending feedback context. Override to store the context.

        Args:
            flow_uuid: Unique identifier for the flow instance
            context: The pending feedback context with all resume information
            state_data: Current state data
        """
        # Default: just save the state without pending context
        self.save_state(flow_uuid, context.method_name, state_data)

    def load_pending_feedback(
        self,
        flow_uuid: str,
    ) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
        """Load state and pending feedback context.

        This method is called when resuming a paused flow. Override to
        load both the state and the pending feedback context.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            Tuple of (state_data, pending_context) if pending feedback exists,
            None otherwise.
        """
        return None

    def clear_pending_feedback(self, flow_uuid: str) -> None:  # noqa: B027
        """Clear the pending feedback marker after successful resume.

        This is called after feedback is received and the flow resumes.
        Optional override to remove the pending feedback marker.

        Args:
            flow_uuid: Unique identifier for the flow instance
        """
        pass
