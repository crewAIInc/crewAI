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
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

from pydantic import BaseModel

from crewai.flow.persistence.base import FlowPersistence

logger = logging.getLogger(__name__)
T = TypeVar("T")


def persist(persistence: FlowPersistence):
    """Decorator to persist flow state after method execution.
    
    This decorator supports both synchronous and asynchronous methods. It will
    persist the flow state after the method completes successfully. For async
    methods, it ensures the state is persisted before returning the result.
    
    Args:
        persistence: FlowPersistence implementation to use for storing state
    
    Returns:
        A decorator function that wraps flow methods and handles state persistence
    
    Raises:
        ValueError: If the flow state doesn't have an 'id' field
        RuntimeError: If state persistence fails
    """
    def _persist_state(flow_instance: Any, method_name: str) -> None:
        """Helper to persist state with error handling."""
        try:
            # Get flow UUID from state
            state = getattr(flow_instance, 'state', None)
            if state is None:
                raise ValueError("Flow instance has no state")
                
            flow_uuid: Optional[str] = None
            if isinstance(state, dict):
                flow_uuid = state.get('id')
            elif isinstance(state, BaseModel):
                flow_uuid = getattr(state, 'id', None)
                
            if not flow_uuid:
                raise ValueError(
                    "Flow state must have an 'id' field for persistence"
                )
                
            # Persist the state
            persistence.save_state(
                flow_uuid=flow_uuid,
                method_name=method_name,
                state_data=state,
            )
        except Exception as e:
            logger.error(
                f"Failed to persist state for method {method_name}: {str(e)}"
            )
            raise RuntimeError(f"State persistence failed: {str(e)}") from e
    
    def decorator(method: Callable[..., T]) -> Callable[..., T]:
        """Decorator that handles both sync and async methods."""
        if asyncio.iscoroutinefunction(method):
            @functools.wraps(method)
            async def async_wrapper(flow_instance: Any, *args: Any, **kwargs: Any) -> T:
                # Execute the original async method
                result = await method(flow_instance, *args, **kwargs)
                # Persist state after method completion
                _persist_state(flow_instance, method.__name__)
                return result
            return cast(Callable[..., T], async_wrapper)
        else:
            @functools.wraps(method)
            def sync_wrapper(flow_instance: Any, *args: Any, **kwargs: Any) -> T:
                # Execute the original sync method
                result = method(flow_instance, *args, **kwargs)
                # Persist state after method completion
                _persist_state(flow_instance, method.__name__)
                return result
            return cast(Callable[..., T], sync_wrapper)
            
    return decorator
