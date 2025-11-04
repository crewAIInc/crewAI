"""Base classes for LLM transport interceptors.

This module provides abstract base classes for intercepting and modifying
outbound and inbound messages at the transport level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


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
