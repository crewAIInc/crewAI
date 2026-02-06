"""A2A error codes and error response utilities.

This module provides a centralized mapping of all A2A protocol error codes
as defined in the A2A specification, plus custom CrewAI extensions.

Error codes follow JSON-RPC 2.0 conventions:
- -32700 to -32600: Standard JSON-RPC errors
- -32099 to -32000: Server errors (A2A-specific)
- -32768 to -32100: Reserved for implementation-defined errors
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from a2a.client.errors import A2AClientTimeoutError


class A2APollingTimeoutError(A2AClientTimeoutError):
    """Raised when polling exceeds the configured timeout."""


class A2AErrorCode(IntEnum):
    """A2A protocol error codes.

    Codes follow JSON-RPC 2.0 specification with A2A-specific extensions.
    """

    # JSON-RPC 2.0 Standard Errors (-32700 to -32600)
    JSON_PARSE_ERROR = -32700
    """Invalid JSON was received by the server."""

    INVALID_REQUEST = -32600
    """The JSON sent is not a valid Request object."""

    METHOD_NOT_FOUND = -32601
    """The method does not exist / is not available."""

    INVALID_PARAMS = -32602
    """Invalid method parameter(s)."""

    INTERNAL_ERROR = -32603
    """Internal JSON-RPC error."""

    # A2A-Specific Errors (-32099 to -32000)
    TASK_NOT_FOUND = -32001
    """The specified task was not found."""

    TASK_NOT_CANCELABLE = -32002
    """The task cannot be canceled (already completed/failed)."""

    PUSH_NOTIFICATION_NOT_SUPPORTED = -32003
    """Push notifications are not supported by this agent."""

    UNSUPPORTED_OPERATION = -32004
    """The requested operation is not supported."""

    CONTENT_TYPE_NOT_SUPPORTED = -32005
    """Incompatible content types between client and server."""

    INVALID_AGENT_RESPONSE = -32006
    """The agent produced an invalid response."""

    # CrewAI Custom Extensions (-32768 to -32100)
    UNSUPPORTED_VERSION = -32009
    """The requested A2A protocol version is not supported."""

    UNSUPPORTED_EXTENSION = -32010
    """Client does not support required protocol extensions."""

    AUTHENTICATION_REQUIRED = -32011
    """Authentication is required for this operation."""

    AUTHORIZATION_FAILED = -32012
    """Authorization check failed (insufficient permissions)."""

    RATE_LIMIT_EXCEEDED = -32013
    """Rate limit exceeded for this client/operation."""

    TASK_TIMEOUT = -32014
    """Task execution timed out."""

    TRANSPORT_NEGOTIATION_FAILED = -32015
    """Failed to negotiate a compatible transport protocol."""

    CONTEXT_NOT_FOUND = -32016
    """The specified context was not found."""

    SKILL_NOT_FOUND = -32017
    """The specified skill was not found."""

    ARTIFACT_NOT_FOUND = -32018
    """The specified artifact was not found."""


# Error code to default message mapping
ERROR_MESSAGES: dict[int, str] = {
    A2AErrorCode.JSON_PARSE_ERROR: "Parse error",
    A2AErrorCode.INVALID_REQUEST: "Invalid Request",
    A2AErrorCode.METHOD_NOT_FOUND: "Method not found",
    A2AErrorCode.INVALID_PARAMS: "Invalid params",
    A2AErrorCode.INTERNAL_ERROR: "Internal error",
    A2AErrorCode.TASK_NOT_FOUND: "Task not found",
    A2AErrorCode.TASK_NOT_CANCELABLE: "Task not cancelable",
    A2AErrorCode.PUSH_NOTIFICATION_NOT_SUPPORTED: "Push Notification is not supported",
    A2AErrorCode.UNSUPPORTED_OPERATION: "This operation is not supported",
    A2AErrorCode.CONTENT_TYPE_NOT_SUPPORTED: "Incompatible content types",
    A2AErrorCode.INVALID_AGENT_RESPONSE: "Invalid agent response",
    A2AErrorCode.UNSUPPORTED_VERSION: "Unsupported A2A version",
    A2AErrorCode.UNSUPPORTED_EXTENSION: "Client does not support required extensions",
    A2AErrorCode.AUTHENTICATION_REQUIRED: "Authentication required",
    A2AErrorCode.AUTHORIZATION_FAILED: "Authorization failed",
    A2AErrorCode.RATE_LIMIT_EXCEEDED: "Rate limit exceeded",
    A2AErrorCode.TASK_TIMEOUT: "Task execution timed out",
    A2AErrorCode.TRANSPORT_NEGOTIATION_FAILED: "Transport negotiation failed",
    A2AErrorCode.CONTEXT_NOT_FOUND: "Context not found",
    A2AErrorCode.SKILL_NOT_FOUND: "Skill not found",
    A2AErrorCode.ARTIFACT_NOT_FOUND: "Artifact not found",
}


@dataclass
class A2AError(Exception):
    """Base exception for A2A protocol errors.

    Attributes:
        code: The A2A/JSON-RPC error code.
        message: Human-readable error message.
        data: Optional additional error data.
    """

    code: int
    message: str | None = None
    data: Any = None

    def __post_init__(self) -> None:
        if self.message is None:
            self.message = ERROR_MESSAGES.get(self.code, "Unknown error")
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-RPC error object format."""
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data is not None:
            error["data"] = self.data
        return error

    def to_response(self, request_id: str | int | None = None) -> dict[str, Any]:
        """Convert to full JSON-RPC error response."""
        return {
            "jsonrpc": "2.0",
            "error": self.to_dict(),
            "id": request_id,
        }


@dataclass
class JSONParseError(A2AError):
    """Invalid JSON was received."""

    code: int = field(default=A2AErrorCode.JSON_PARSE_ERROR, init=False)


@dataclass
class InvalidRequestError(A2AError):
    """The JSON sent is not a valid Request object."""

    code: int = field(default=A2AErrorCode.INVALID_REQUEST, init=False)


@dataclass
class MethodNotFoundError(A2AError):
    """The method does not exist / is not available."""

    code: int = field(default=A2AErrorCode.METHOD_NOT_FOUND, init=False)
    method: str | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.method:
            self.message = f"Method not found: {self.method}"
        super().__post_init__()


@dataclass
class InvalidParamsError(A2AError):
    """Invalid method parameter(s)."""

    code: int = field(default=A2AErrorCode.INVALID_PARAMS, init=False)
    param: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.message is None:
            if self.param and self.reason:
                self.message = f"Invalid parameter '{self.param}': {self.reason}"
            elif self.param:
                self.message = f"Invalid parameter: {self.param}"
        super().__post_init__()


@dataclass
class InternalError(A2AError):
    """Internal JSON-RPC error."""

    code: int = field(default=A2AErrorCode.INTERNAL_ERROR, init=False)


@dataclass
class TaskNotFoundError(A2AError):
    """The specified task was not found."""

    code: int = field(default=A2AErrorCode.TASK_NOT_FOUND, init=False)
    task_id: str | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.task_id:
            self.message = f"Task not found: {self.task_id}"
        super().__post_init__()


@dataclass
class TaskNotCancelableError(A2AError):
    """The task cannot be canceled."""

    code: int = field(default=A2AErrorCode.TASK_NOT_CANCELABLE, init=False)
    task_id: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.message is None:
            if self.task_id and self.reason:
                self.message = f"Task {self.task_id} cannot be canceled: {self.reason}"
            elif self.task_id:
                self.message = f"Task {self.task_id} cannot be canceled"
        super().__post_init__()


@dataclass
class PushNotificationNotSupportedError(A2AError):
    """Push notifications are not supported."""

    code: int = field(default=A2AErrorCode.PUSH_NOTIFICATION_NOT_SUPPORTED, init=False)


@dataclass
class UnsupportedOperationError(A2AError):
    """The requested operation is not supported."""

    code: int = field(default=A2AErrorCode.UNSUPPORTED_OPERATION, init=False)
    operation: str | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.operation:
            self.message = f"Operation not supported: {self.operation}"
        super().__post_init__()


@dataclass
class ContentTypeNotSupportedError(A2AError):
    """Incompatible content types."""

    code: int = field(default=A2AErrorCode.CONTENT_TYPE_NOT_SUPPORTED, init=False)
    requested_types: list[str] | None = None
    supported_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.requested_types and self.supported_types:
            self.message = (
                f"Content type not supported. Requested: {self.requested_types}, "
                f"Supported: {self.supported_types}"
            )
        super().__post_init__()


@dataclass
class InvalidAgentResponseError(A2AError):
    """The agent produced an invalid response."""

    code: int = field(default=A2AErrorCode.INVALID_AGENT_RESPONSE, init=False)


@dataclass
class UnsupportedVersionError(A2AError):
    """The requested A2A version is not supported."""

    code: int = field(default=A2AErrorCode.UNSUPPORTED_VERSION, init=False)
    requested_version: str | None = None
    supported_versions: list[str] | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.requested_version:
            msg = f"Unsupported A2A version: {self.requested_version}"
            if self.supported_versions:
                msg += f". Supported versions: {', '.join(self.supported_versions)}"
            self.message = msg
        super().__post_init__()


@dataclass
class UnsupportedExtensionError(A2AError):
    """Client does not support required extensions."""

    code: int = field(default=A2AErrorCode.UNSUPPORTED_EXTENSION, init=False)
    required_extensions: list[str] | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.required_extensions:
            self.message = f"Client does not support required extensions: {', '.join(self.required_extensions)}"
        super().__post_init__()


@dataclass
class AuthenticationRequiredError(A2AError):
    """Authentication is required."""

    code: int = field(default=A2AErrorCode.AUTHENTICATION_REQUIRED, init=False)


@dataclass
class AuthorizationFailedError(A2AError):
    """Authorization check failed."""

    code: int = field(default=A2AErrorCode.AUTHORIZATION_FAILED, init=False)
    required_scope: str | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.required_scope:
            self.message = (
                f"Authorization failed. Required scope: {self.required_scope}"
            )
        super().__post_init__()


@dataclass
class RateLimitExceededError(A2AError):
    """Rate limit exceeded."""

    code: int = field(default=A2AErrorCode.RATE_LIMIT_EXCEEDED, init=False)
    retry_after: int | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.retry_after:
            self.message = (
                f"Rate limit exceeded. Retry after {self.retry_after} seconds"
            )
        if self.retry_after:
            self.data = {"retry_after": self.retry_after}
        super().__post_init__()


@dataclass
class TaskTimeoutError(A2AError):
    """Task execution timed out."""

    code: int = field(default=A2AErrorCode.TASK_TIMEOUT, init=False)
    task_id: str | None = None
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.message is None:
            if self.task_id and self.timeout_seconds:
                self.message = (
                    f"Task {self.task_id} timed out after {self.timeout_seconds}s"
                )
            elif self.task_id:
                self.message = f"Task {self.task_id} timed out"
        super().__post_init__()


@dataclass
class TransportNegotiationFailedError(A2AError):
    """Failed to negotiate a compatible transport protocol."""

    code: int = field(default=A2AErrorCode.TRANSPORT_NEGOTIATION_FAILED, init=False)
    client_transports: list[str] | None = None
    server_transports: list[str] | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.client_transports and self.server_transports:
            self.message = (
                f"Transport negotiation failed. Client: {self.client_transports}, "
                f"Server: {self.server_transports}"
            )
        super().__post_init__()


@dataclass
class ContextNotFoundError(A2AError):
    """The specified context was not found."""

    code: int = field(default=A2AErrorCode.CONTEXT_NOT_FOUND, init=False)
    context_id: str | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.context_id:
            self.message = f"Context not found: {self.context_id}"
        super().__post_init__()


@dataclass
class SkillNotFoundError(A2AError):
    """The specified skill was not found."""

    code: int = field(default=A2AErrorCode.SKILL_NOT_FOUND, init=False)
    skill_id: str | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.skill_id:
            self.message = f"Skill not found: {self.skill_id}"
        super().__post_init__()


@dataclass
class ArtifactNotFoundError(A2AError):
    """The specified artifact was not found."""

    code: int = field(default=A2AErrorCode.ARTIFACT_NOT_FOUND, init=False)
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        if self.message is None and self.artifact_id:
            self.message = f"Artifact not found: {self.artifact_id}"
        super().__post_init__()


def create_error_response(
    code: int | A2AErrorCode,
    message: str | None = None,
    data: Any = None,
    request_id: str | int | None = None,
) -> dict[str, Any]:
    """Create a JSON-RPC error response.

    Args:
        code: Error code (A2AErrorCode or int).
        message: Optional error message (uses default if not provided).
        data: Optional additional error data.
        request_id: Request ID for correlation.

    Returns:
        Dict in JSON-RPC error response format.
    """
    error = A2AError(code=int(code), message=message, data=data)
    return error.to_response(request_id)


def is_retryable_error(code: int) -> bool:
    """Check if an error is potentially retryable.

    Args:
        code: Error code to check.

    Returns:
        True if the error might be resolved by retrying.
    """
    retryable_codes = {
        A2AErrorCode.INTERNAL_ERROR,
        A2AErrorCode.RATE_LIMIT_EXCEEDED,
        A2AErrorCode.TASK_TIMEOUT,
    }
    return code in retryable_codes


def is_client_error(code: int) -> bool:
    """Check if an error is a client-side error.

    Args:
        code: Error code to check.

    Returns:
        True if the error is due to client request issues.
    """
    client_error_codes = {
        A2AErrorCode.JSON_PARSE_ERROR,
        A2AErrorCode.INVALID_REQUEST,
        A2AErrorCode.METHOD_NOT_FOUND,
        A2AErrorCode.INVALID_PARAMS,
        A2AErrorCode.TASK_NOT_FOUND,
        A2AErrorCode.CONTENT_TYPE_NOT_SUPPORTED,
        A2AErrorCode.UNSUPPORTED_VERSION,
        A2AErrorCode.UNSUPPORTED_EXTENSION,
        A2AErrorCode.CONTEXT_NOT_FOUND,
        A2AErrorCode.SKILL_NOT_FOUND,
        A2AErrorCode.ARTIFACT_NOT_FOUND,
    }
    return code in client_error_codes
