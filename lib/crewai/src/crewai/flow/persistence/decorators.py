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

from collections.abc import Awaitable, Callable
import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any, Final, ParamSpec, TypeVar, cast

from crewai_core.printer import PRINTER
from pydantic import BaseModel

from crewai.flow.flow_wrappers import FlowMethod
from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


if TYPE_CHECKING:
    from crewai.flow.flow import Flow


logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

# Constants for log messages
LOG_MESSAGES: Final[dict[str, str]] = {
    "save_state": "Saving flow state to memory for ID: {}",
    "save_error": "Failed to persist state for method {}: {}",
    "state_missing": "Flow instance has no state",
    "id_missing": "Flow state must have an 'id' field for persistence",
}


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

            # Log state saving only if verbose is True
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


class PersistedFlowMethod(FlowMethod[P, R]):
    """FlowMethod variant that persists state after each invocation.

    Wrapping the original method directly (rather than copying its attributes
    onto a closure) lets ``FlowMethod.__getattr__`` delegate flow flags like
    ``__is_start_method__`` to the wrapped object transparently.

    For async wrapped methods, ``R`` is the ``Coroutine`` returned by calling
    them, so ``__call__``'s declared return type stays accurate in both cases.
    """

    def __init__(
        self,
        meth: Callable[P, R],
        instance: Any = None,
        *,
        persistence: FlowPersistence | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(meth, instance)
        self._persistence = persistence
        self._verbose = verbose

    def _resolve_flow_instance(self, args: tuple[Any, ...]) -> Any:
        return (
            self._instance
            if self._instance is not None
            else (args[0] if args else None)
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        if inspect.iscoroutinefunction(self._meth):
            return cast(R, self._call_async(*args, **kwargs))
        flow_instance = self._resolve_flow_instance(args)
        result = super().__call__(*args, **kwargs)
        PersistenceDecorator.persist_state(
            flow_instance,
            self.__name__,
            cast(FlowPersistence, self._persistence),
            self._verbose,
        )
        return result

    async def _call_async(self, *args: Any, **kwargs: Any) -> Any:
        flow_instance = self._resolve_flow_instance(args)
        meth = cast(Callable[..., Awaitable[Any]], self._meth)
        if self._instance is not None:
            result = await meth(self._instance, *args, **kwargs)
        else:
            result = await meth(*args, **kwargs)
        PersistenceDecorator.persist_state(
            flow_instance,
            self.__name__,
            cast(FlowPersistence, self._persistence),
            self._verbose,
        )
        return result


def persist(
    persistence: FlowPersistence | None = None, verbose: bool = False
) -> Callable[
    [type[Flow[Any]] | Callable[..., T]],
    type[Flow[Any]] | Callable[..., T],
]:
    """Decorator to persist flow state.

    This decorator can be applied at either the class level or method level.
    When applied at the class level, it automatically persists all flow method
    states. When applied at the method level, it persists only that method's
    state.

    Args:
        persistence: Optional FlowPersistence implementation to use.
                    If not provided, uses SQLiteFlowPersistence.
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

    def decorator(
        target: type[Flow[Any]] | Callable[..., T],
    ) -> type[Flow[Any]] | Callable[..., T]:
        """Decorator that handles both class and method decoration."""
        actual_persistence = persistence or SQLiteFlowPersistence()

        if isinstance(target, type):
            original_init = target.__init__  # type: ignore[misc]

            @functools.wraps(original_init)
            def new_init(self: Flow[Any], *args: Any, **kwargs: Any) -> None:
                if "persistence" not in kwargs:
                    kwargs["persistence"] = actual_persistence
                original_init(self, *args, **kwargs)

            target.__init__ = new_init  # type: ignore[misc]

            for name, method in list(target.__dict__.items()):
                if not isinstance(method, FlowMethod):
                    continue
                setattr(
                    target,
                    name,
                    PersistedFlowMethod(
                        method, persistence=actual_persistence, verbose=verbose
                    ),
                )

            return target

        return cast(
            Callable[..., T],
            PersistedFlowMethod(
                target, persistence=actual_persistence, verbose=verbose
            ),
        )

    return decorator
