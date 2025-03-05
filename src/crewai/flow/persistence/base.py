"""Base class for flow state persistence."""

import abc
from typing import Any, Dict, Optional, Union

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
        pass

    @abc.abstractmethod
    def save_state(
        self,
        flow_uuid: str,
        method_name: str,
        state_data: Union[Dict[str, Any], BaseModel],
    ) -> None:
        """Persist the flow state after method completion.

        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
            state_data: Current state data (either dict or Pydantic model)
        """
        pass

    @abc.abstractmethod
    def load_state(self, flow_uuid: str) -> Optional[Dict[str, Any]]:
        """Load the most recent state for a given flow UUID.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            The most recent state as a dictionary, or None if no state exists
        """
        pass

    def _convert_to_dict(self, obj: Any) -> Any:
        """Recursively convert Pydantic models to dictionaries.

        This helper method ensures all Pydantic models in the state are
        properly converted to dictionaries for JSON serialization.

        Args:
            obj: The object to convert

        Returns:
            A JSON-serializable version of the object
        """
        if isinstance(obj, BaseModel):
            # Convert Pydantic model to dict
            if hasattr(obj, "model_dump"):
                # Pydantic v2
                obj_dict = obj.model_dump()
            else:
                # Pydantic v1
                obj_dict = obj.dict()
            # Recursively convert any nested models
            return {k: self._convert_to_dict(v) for k, v in obj_dict.items()}
        elif isinstance(obj, dict):
            # Recursively convert dict values
            return {k: self._convert_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Recursively convert list items
            return [self._convert_to_dict(item) for item in obj]
        elif isinstance(obj, tuple):
            # Recursively convert tuple items
            return tuple(self._convert_to_dict(item) for item in obj)
        elif isinstance(obj, set):
            # Recursively convert set items
            return {self._convert_to_dict(item) for item in obj}
        else:
            # Return primitive types as is
            return obj
