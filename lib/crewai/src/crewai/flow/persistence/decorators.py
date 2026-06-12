"""
Decorators for flow state persistence.

Example:
    ```python
    from crewai.flow.flow import Flow, start
    from crewai.flow.persistence import persist, SQLiteFlowPersistence


    class MyFlow(Flow):
        @start()
        @persist(SQLiteFlowPersistence())
        def sync_method(self):
            # Synchronous method implementation
            pass

        @start()
        @persist(SQLiteFlowPersistence())
        async def async_method(self):
            # Asynchronous method implementation
            await some_async_operation()
    ```
"""

from __future__ import annotations

from collections.abc import Callable
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final, TypeVar

from crewai_core.printer import PRINTER
from pydantic import BaseModel

from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.factory import default_flow_persistence


if TYPE_CHECKING:
    from crewai.flow.runtime import Flow


logger = logging.getLogger(__name__)
T = TypeVar("T")

__all__ = ["PersistenceDecorator", "persist"]

LOG_MESSAGES: Final[dict[str, str]] = {
    "save_state": "Saving flow state to memory for ID: {}",
    "save_error": "Failed to persist state for method {}: {}",
    "state_missing": "Flow instance has no state",
    "id_missing": "Flow state must have an 'id' field for persistence",
}


def _stamp_persistence_metadata(
    target: Any,
    persistence: FlowPersistence,
    verbose: bool,
) -> None:
    target.__flow_persistence_config__ = SimpleNamespace(
        persistence=persistence,
        verbose=verbose,
    )


class PersistenceDecorator:
    """Class to handle flow state persistence with consistent logging."""

    @classmethod
    def persist_state(
        cls,
        flow_instance: Flow[Any],
        method_name: str,
        persistence_instance: FlowPersistence,
        verbose: bool = False,
    ) -> None:
        """Persist flow state with proper error handling and logging.

        This method handles the persistence of flow state data, including proper
        error handling and colored console output for status updates.

        Args:
            flow_instance: The flow instance whose state to persist
            method_name: Name of the method that triggered persistence
            persistence_instance: The persistence backend to use
            verbose: Whether to log persistence operations

        Raises:
            ValueError: If flow has no state or state lacks an ID
            RuntimeError: If state persistence fails
            AttributeError: If flow instance lacks required state attributes
        """
        try:
            state = getattr(flow_instance, "state", None)
            if state is None:
                raise ValueError("Flow instance has no state")

            flow_uuid: str | None = None
            if isinstance(state, dict):
                flow_uuid = state.get("id")
            elif hasattr(state, "_unwrap"):
                unwrapped = state._unwrap()
                if isinstance(unwrapped, dict):
                    flow_uuid = unwrapped.get("id")
                else:
                    flow_uuid = getattr(unwrapped, "id", None)
            elif isinstance(state, BaseModel) or hasattr(state, "id"):
                flow_uuid = getattr(state, "id", None)

            if not flow_uuid:
                raise ValueError("Flow state must have an 'id' field for persistence")

            if verbose:
                PRINTER.print(
                    LOG_MESSAGES["save_state"].format(flow_uuid), color="cyan"
                )
                logger.info(LOG_MESSAGES["save_state"].format(flow_uuid))

            try:
                state_data = state._unwrap() if hasattr(state, "_unwrap") else state
                persistence_instance.save_state(
                    flow_uuid=flow_uuid,
                    method_name=method_name,
                    state_data=state_data,
                )
            except Exception as e:
                error_msg = LOG_MESSAGES["save_error"].format(method_name, str(e))
                if verbose:
                    PRINTER.print(error_msg, color="red")
                logger.error(error_msg)
                raise RuntimeError(f"State persistence failed: {e!s}") from e
        except AttributeError as e:
            error_msg = LOG_MESSAGES["state_missing"]
            if verbose:
                PRINTER.print(error_msg, color="red")
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        except (TypeError, ValueError) as e:
            error_msg = LOG_MESSAGES["id_missing"]
            if verbose:
                PRINTER.print(error_msg, color="red")
            logger.error(error_msg)
            raise ValueError(error_msg) from e


def persist(
    persistence: FlowPersistence | None = None, verbose: bool = False
) -> Callable[[type | Callable[..., T]], type | Callable[..., T]]:
    """Decorator to persist flow state.

    This decorator can be applied at either the class level or method level.
    When applied at the class level, it automatically persists all flow method
    states. When applied at the method level, it persists only that method's
    state.

    The decorator is a pure metadata stamper: it records the persistence
    configuration on the class or method, and the Flow engine saves state
    after each persisted method completes, driven by the flow's definition.

    Args:
        persistence: Optional FlowPersistence implementation to use.
                    If not provided, uses ``default_flow_persistence()`` (the
                    registered factory when present, else the built-in SQLite
                    fallback).
        verbose: Whether to log persistence operations. Defaults to False.

    Returns:
        A decorator that can be applied to either a class or method

    Raises:
        ValueError: If the flow state doesn't have an 'id' field
        RuntimeError: If state persistence fails

    Example:
        @persist(verbose=True)  # Class-level persistence with logging
        class MyFlow(Flow[MyState]):
            @start()
            def begin(self):
                pass
    """

    def decorator(target: type | Callable[..., T]) -> type | Callable[..., T]:
        actual_persistence = (
            persistence if persistence is not None else default_flow_persistence()
        )

        _stamp_persistence_metadata(target, actual_persistence, verbose)
        return target

    return decorator
