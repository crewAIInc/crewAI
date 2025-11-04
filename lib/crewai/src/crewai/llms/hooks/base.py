"""Base classes for LLM transport interceptors.

This module provides abstract base classes for intercepting and modifying
outbound and inbound messages at the transport level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


class BaseInterceptor(ABC, Generic[T]):
    """Abstract base class for intercepting transport-level messages.

    This class provides hooks to intercept and modify outbound and inbound
    messages before they are sent to or after they are received from LLM providers.

    The generic type parameter MessageT ensures that both on_outbound and on_inbound
    methods work with the same message type, maintaining type safety throughout
    the interception chain.

    Use cases include:
    - Custom header injection
    - Request/response logging
    - URL modification
    - Authentication schemes
    - Message transformation
    - Proxy integration

    Example:
        >>> class CustomInterceptor(BaseInterceptor[httpx.Request]):
        ...     def on_outbound(self, message: httpx.Request) -> httpx.Request:
        ...         # Modify headers, URL, or body
        ...         message.headers["X-Custom-Header"] = "value"
        ...         return message
        ...
        ...     def on_inbound(self, message: httpx.Response) -> httpx.Response:
        ...         # Transform or log inbound message
        ...         print(f"Status: {message.status_code}")
        ...         return message
        ...
        ...     async def aon_outbound(self, message: httpx.Request) -> httpx.Request:
        ...         # Async version for async transports
        ...         message.headers["X-Custom-Header"] = "value"
        ...         return message
        ...
        ...     async def aon_inbound(self, message: httpx.Response) -> httpx.Response:
        ...         # Async version for async transports
        ...         print(f"Status: {message.status_code}")
        ...         return message
    """

    @abstractmethod
    def on_outbound(self, message: T) -> T:
        """Modify outbound message before it's sent to the LLM provider.

        This method is called just before the message is sent, allowing you
        to inspect and modify the message object.

        Args:
            message: The outbound message object. Type varies by provider:
                - OpenAI/Anthropic: httpx.Request
                - Google/Gemini: requests.Request
                - Azure: requests.Request or httpx.Request
                - AWS Bedrock: botocore request dict

        Returns:
            Modified message object of the same type that will be sent to the provider.

        Note:
            Common modifications include:
            - message.headers: Add/modify headers
            - message.url: Modify the request URL
            - message.content/body: Modify message body
            - message.method: Change HTTP method (use with caution)
        """
        ...

    @abstractmethod
    def on_inbound(self, message: T) -> T:
        """Modify inbound message after receiving it from the LLM provider.

        This method is called immediately after receiving the inbound message,
        allowing you to inspect and transform the message object.

        Args:
            message: The inbound message object. Type varies by provider:
                - OpenAI/Anthropic: httpx.Response
                - Google/Gemini: requests.Response
                - Azure: requests.Response or httpx.Response
                - AWS Bedrock: botocore response dict

        Returns:
            Modified message object of the same type that will be processed by the SDK.

        Note:
            Common operations include:
            - message.status_code: Check status
            - message.headers: Inspect headers
            - message.content/text/json(): Access message body
            - Logging, metrics collection, error handling
        """
        ...

    async def aon_outbound(self, message: T) -> T:
        """Async version of on_outbound for async transports.

        This method is called just before the message is sent in async contexts,
        allowing you to inspect and modify the message object asynchronously.

        Args:
            message: The outbound message object. Type varies by provider:
                - OpenAI/Anthropic: httpx.Request
                - Google/Gemini: requests.Request
                - Azure: requests.Request or httpx.Request
                - AWS Bedrock: botocore request dict

        Returns:
            Modified message object of the same type that will be sent to the provider.

        Note:
            Common modifications include:
            - message.headers: Add/modify headers
            - message.url: Modify the request URL
            - message.content/body: Modify message body
            - message.method: Change HTTP method (use with caution)
        """
        raise NotImplementedError

    async def aon_inbound(self, message: T) -> T:
        """Async version of on_inbound for async transports.

        This method is called immediately after receiving the inbound message
        in async contexts, allowing you to inspect and transform the message object.

        Args:
            message: The inbound message object. Type varies by provider:
                - OpenAI/Anthropic: httpx.Response
                - Google/Gemini: requests.Response
                - Azure: requests.Response or httpx.Response
                - AWS Bedrock: botocore response dict

        Returns:
            Modified message object of the same type that will be processed by the SDK.

        Note:
            Common operations include:
            - message.status_code: Check status
            - message.headers: Inspect headers
            - message.content/text/json(): Access message body
            - Logging, metrics collection, error handling
        """
        raise NotImplementedError
