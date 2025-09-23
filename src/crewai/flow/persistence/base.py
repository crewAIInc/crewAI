"""Base class for flow state persistence."""

import abc
from typing import Any

from pydantic import BaseModel


class FlowPersistence(abc.ABC):
    """Abstract base class for flow state persistence.

    This class defines the interface that all persistence implementations must follow.
    It supports both structured (Pydantic BaseModel) and unstructured (dict) states.
    """

    @abc.abstractmethod
    def init_db(self) -> None:
        """Initialize the persistence backend.

        This method should handle any necessary setup, such as:
        - Creating tables
        - Establishing connections
        - Setting up indexes
        """

    @abc.abstractmethod
    def save_state(
        self, flow_uuid: str, method_name: str, state_data: dict[str, Any] | BaseModel
    ) -> None:
        """Persist the flow state after method completion.

        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
            state_data: Current state data (either dict or Pydantic model)
        """

    @abc.abstractmethod
    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        """Load the most recent state for a given flow UUID.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            The most recent state as a dictionary, or None if no state exists
        """
