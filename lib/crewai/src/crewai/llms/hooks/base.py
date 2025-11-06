"""Base classes for LLM transport interceptors.

This module provides abstract base classes for intercepting and modifying
outbound and inbound messages at the transport level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic_core import core_schema


if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema


T = TypeVar("T")
U = TypeVar("U")


class BaseInterceptor(ABC, Generic[T, U]):
    """Abstract base class for intercepting transport-level messages.

    Provides hooks to intercept and modify outbound and inbound messages
    at the transport layer.

    Type parameters:
        T: Outbound message type (e.g., httpx.Request)
        U: Inbound message type (e.g., httpx.Response)

    Example:
        >>> import httpx
        >>> class CustomInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
        ...     def on_outbound(self, message: httpx.Request) -> httpx.Request:
        ...         message.headers["X-Custom-Header"] = "value"
        ...         return message
        ...
        ...     def on_inbound(self, message: httpx.Response) -> httpx.Response:
        ...         print(f"Status: {message.status_code}")
        ...         return message
    """

    @abstractmethod
    def on_outbound(self, message: T) -> T:
        """Intercept outbound message before sending.

        Args:
            message: Outbound message object.

        Returns:
            Modified message object.
        """
        ...

    @abstractmethod
    def on_inbound(self, message: U) -> U:
        """Intercept inbound message after receiving.

        Args:
            message: Inbound message object.

        Returns:
            Modified message object.
        """
        ...

    async def aon_outbound(self, message: T) -> T:
        """Async version of on_outbound.

        Args:
            message: Outbound message object.

        Returns:
            Modified message object.
        """
        raise NotImplementedError

    async def aon_inbound(self, message: U) -> U:
        """Async version of on_inbound.

        Args:
            message: Inbound message object.

        Returns:
            Modified message object.
        """
        raise NotImplementedError

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Generate Pydantic core schema for BaseInterceptor.

        This allows the generic BaseInterceptor to be used in Pydantic models
        without requiring arbitrary_types_allowed=True. The schema validates
        that the value is an instance of BaseInterceptor.

        Args:
            _source_type: The source type being validated (unused).
            _handler: Handler for generating schemas (unused).

        Returns:
            A Pydantic core schema that validates BaseInterceptor instances.
        """
        return core_schema.no_info_plain_validator_function(
            _validate_interceptor,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: x, return_schema=core_schema.any_schema()
            ),
        )


def _validate_interceptor(value: Any) -> BaseInterceptor[T, U]:
    """Validate that the value is a BaseInterceptor instance.

    Args:
        value: The value to validate.

    Returns:
        The validated BaseInterceptor instance.

    Raises:
        ValueError: If the value is not a BaseInterceptor instance.
    """
    if not isinstance(value, BaseInterceptor):
        raise ValueError(
            f"Expected BaseInterceptor instance, got {type(value).__name__}"
        )
    return value
