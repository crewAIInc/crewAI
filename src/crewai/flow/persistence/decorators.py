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

import asyncio
import functools
import logging
from typing import (
    Any,
    Callable,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel

from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from crewai.utilities.printer import Printer

logger = logging.getLogger(__name__)
T = TypeVar("T")

# Constants for log messages
LOG_MESSAGES = {
    "save_state": "Saving flow state to memory for ID: {}",
    "save_error": "Failed to persist state for method {}: {}",
    "state_missing": "Flow instance has no state",
    "id_missing": "Flow state must have an 'id' field for persistence"
}


class PersistenceDecorator:
    """Class to handle flow state persistence with consistent logging."""
    
    _printer = Printer()  # Class-level printer instance
    
    @classmethod
    def persist_state(cls, flow_instance: Any, method_name: str, persistence_instance: FlowPersistence) -> None:
        """Persist flow state with proper error handling and logging.
        
        This method handles the persistence of flow state data, including proper
        error handling and colored console output for status updates.
        
        Args:
            flow_instance: The flow instance whose state to persist
            method_name: Name of the method that triggered persistence
            persistence_instance: The persistence backend to use
            
        Raises:
            ValueError: If flow has no state or state lacks an ID
            RuntimeError: If state persistence fails
            AttributeError: If flow instance lacks required state attributes
            
        Note:
            Uses bold_yellow color for success messages and red for errors.
            All operations are logged at appropriate levels (info/error).
            
        Example:
            ```python
            @persist
            def my_flow_method(self):
                # Method implementation
                pass
            # State will be automatically persisted after method execution
            ```
        """
        try:
            state = getattr(flow_instance, 'state', None)
            if state is None:
                raise ValueError("Flow instance has no state")
                
            flow_uuid: Optional[str] = None
            if isinstance(state, dict):
                flow_uuid = state.get('id')
            elif isinstance(state, BaseModel):
                flow_uuid = getattr(state, 'id', None)
                
            if not flow_uuid:
                raise ValueError("Flow state must have an 'id' field for persistence")
                
            # Log state saving with consistent message
            cls._printer.print(LOG_MESSAGES["save_state"].format(flow_uuid), color="bold_yellow")
            logger.info(LOG_MESSAGES["save_state"].format(flow_uuid))
            
            try:
                persistence_instance.save_state(
                    flow_uuid=flow_uuid,
                    method_name=method_name,
                    state_data=state,
                )
            except Exception as e:
                error_msg = LOG_MESSAGES["save_error"].format(method_name, str(e))
                cls._printer.print(error_msg, color="red")
                logger.error(error_msg)
                raise RuntimeError(f"State persistence failed: {str(e)}") from e
        except AttributeError:
            error_msg = LOG_MESSAGES["state_missing"]
            cls._printer.print(error_msg, color="red")
            logger.error(error_msg)
            raise ValueError(error_msg)
        except (TypeError, ValueError) as e:
            error_msg = LOG_MESSAGES["id_missing"]
            cls._printer.print(error_msg, color="red")
            logger.error(error_msg)
            raise ValueError(error_msg) from e


def persist(persistence: Optional[FlowPersistence] = None):
    """Decorator to persist flow state.

    This decorator can be applied at either the class level or method level.
    When applied at the class level, it automatically persists all flow method
    states. When applied at the method level, it persists only that method's
    state.

    Args:
        persistence: Optional FlowPersistence implementation to use.
                    If not provided, uses SQLiteFlowPersistence.

    Returns:
        A decorator that can be applied to either a class or method

    Raises:
        ValueError: If the flow state doesn't have an 'id' field
        RuntimeError: If state persistence fails

    Example:
        @persist  # Class-level persistence with default SQLite
        class MyFlow(Flow[MyState]):
            @start()
            def begin(self):
                pass
    """
    def decorator(target: Union[Type, Callable[..., T]]) -> Union[Type, Callable[..., T]]:
        """Decorator that handles both class and method decoration."""
        actual_persistence = persistence or SQLiteFlowPersistence()

        if isinstance(target, type):
            # Class decoration
            class_methods = {}
            for name, method in target.__dict__.items():
                if callable(method) and hasattr(method, "__is_flow_method__"):
                    # Wrap each flow method with persistence
                    if asyncio.iscoroutinefunction(method):
                        @functools.wraps(method)
                        async def class_async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                            method_coro = method(self, *args, **kwargs)
                            if asyncio.iscoroutine(method_coro):
                                result = await method_coro
                            else:
                                result = method_coro
                            PersistenceDecorator.persist_state(self, method.__name__, actual_persistence)
                            return result
                        class_methods[name] = class_async_wrapper
                    else:
                        @functools.wraps(method)
                        def class_sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                            result = method(self, *args, **kwargs)
                            PersistenceDecorator.persist_state(self, method.__name__, actual_persistence)
                            return result
                        class_methods[name] = class_sync_wrapper

                    # Preserve flow-specific attributes
                    for attr in ["__is_start_method__", "__trigger_methods__", "__condition_type__", "__is_router__"]:
                        if hasattr(method, attr):
                            setattr(class_methods[name], attr, getattr(method, attr))
                    setattr(class_methods[name], "__is_flow_method__", True)

            # Update class with wrapped methods
            for name, method in class_methods.items():
                setattr(target, name, method)
            return target
        else:
            # Method decoration
            method = target
            setattr(method, "__is_flow_method__", True)

            if asyncio.iscoroutinefunction(method):
                @functools.wraps(method)
                async def method_async_wrapper(flow_instance: Any, *args: Any, **kwargs: Any) -> T:
                    method_coro = method(flow_instance, *args, **kwargs)
                    if asyncio.iscoroutine(method_coro):
                        result = await method_coro
                    else:
                        result = method_coro
                    PersistenceDecorator.persist_state(flow_instance, method.__name__, actual_persistence)
                    return result
                for attr in ["__is_start_method__", "__trigger_methods__", "__condition_type__", "__is_router__"]:
                    if hasattr(method, attr):
                        setattr(method_async_wrapper, attr, getattr(method, attr))
                setattr(method_async_wrapper, "__is_flow_method__", True)
                return cast(Callable[..., T], method_async_wrapper)
            else:
                @functools.wraps(method)
                def method_sync_wrapper(flow_instance: Any, *args: Any, **kwargs: Any) -> T:
                    result = method(flow_instance, *args, **kwargs)
                    PersistenceDecorator.persist_state(flow_instance, method.__name__, actual_persistence)
                    return result
                for attr in ["__is_start_method__", "__trigger_methods__", "__condition_type__", "__is_router__"]:
                    if hasattr(method, attr):
                        setattr(method_sync_wrapper, attr, getattr(method, attr))
                setattr(method_sync_wrapper, "__is_flow_method__", True)
                return cast(Callable[..., T], method_sync_wrapper)

    return decorator
