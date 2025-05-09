"""
A2A protocol types for CrewAI.

This module implements the A2A (Agent-to-Agent) protocol types as defined by Google.
The A2A protocol enables interoperability between different agent systems.

For more information, see: https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/
"""

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Self, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_serializer, model_validator


class TaskState(str, Enum):
    """Task state in the A2A protocol."""
    SUBMITTED = 'submitted'
    WORKING = 'working'
    INPUT_REQUIRED = 'input-required'
    COMPLETED = 'completed'
    CANCELED = 'canceled'
    FAILED = 'failed'
    UNKNOWN = 'unknown'
    EXPIRED = 'expired'
    
    @classmethod
    def valid_transitions(cls) -> Dict[str, List[str]]:
        """Get valid state transitions.
        
        Returns:
            A dictionary mapping from state to list of valid next states.
        """
        return {
            cls.SUBMITTED: [cls.WORKING, cls.CANCELED, cls.FAILED],
            cls.WORKING: [cls.INPUT_REQUIRED, cls.COMPLETED, cls.CANCELED, cls.FAILED],
            cls.INPUT_REQUIRED: [cls.WORKING, cls.CANCELED, cls.FAILED],
            cls.COMPLETED: [],  # Terminal state
            cls.CANCELED: [],   # Terminal state
            cls.FAILED: [],     # Terminal state
            cls.UNKNOWN: [cls.SUBMITTED, cls.WORKING, cls.INPUT_REQUIRED, cls.COMPLETED, cls.CANCELED, cls.FAILED],
            cls.EXPIRED: [],    # Terminal state
        }
        
    @classmethod
    def is_valid_transition(cls, from_state: 'TaskState', to_state: 'TaskState') -> bool:
        """Check if a state transition is valid.
        
        Args:
            from_state: The current state.
            to_state: The target state.
            
        Returns:
            True if the transition is valid, False otherwise.
        """
        if from_state == to_state:
            return True
            
        valid_next_states = cls.valid_transitions().get(from_state, [])
        return to_state in valid_next_states


class TextPart(BaseModel):
    """Text part in the A2A protocol."""
    type: Literal['text'] = 'text'
    text: str
    metadata: Optional[Dict[str, Any]] = None


class FileContent(BaseModel):
    """File content in the A2A protocol."""
    name: Optional[str] = None
    mimeType: Optional[str] = None
    bytes: Optional[str] = None
    uri: Optional[str] = None

    @model_validator(mode='after')
    def check_content(self) -> Self:
        """Validate file content has either bytes or uri."""
        if not (self.bytes or self.uri):
            raise ValueError(
                "Either 'bytes' or 'uri' must be present in the file data"
            )
        if self.bytes and self.uri:
            raise ValueError(
                "Only one of 'bytes' or 'uri' can be present in the file data"
            )
        return self


class FilePart(BaseModel):
    """File part in the A2A protocol."""
    type: Literal['file'] = 'file'
    file: FileContent
    metadata: Optional[Dict[str, Any]] = None


class DataPart(BaseModel):
    """Data part in the A2A protocol."""
    type: Literal['data'] = 'data'
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


Part = Annotated[Union[TextPart, FilePart, DataPart], Field(discriminator='type')]


class Message(BaseModel):
    """Message in the A2A protocol."""
    role: Literal['user', 'agent']
    parts: List[Part]
    metadata: Optional[Dict[str, Any]] = None


class TaskStatus(BaseModel):
    """Task status in the A2A protocol."""
    state: TaskState
    message: Optional[Message] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    previous_state: Optional[TaskState] = None

    @field_serializer('timestamp')
    def serialize_dt(self, dt: datetime, _info):
        """Serialize datetime to ISO format."""
        return dt.isoformat()
        
    @model_validator(mode='after')
    def validate_state_transition(self) -> Self:
        """Validate state transition."""
        if self.previous_state and not TaskState.is_valid_transition(self.previous_state, self.state):
            raise ValueError(
                f"Invalid state transition from {self.previous_state} to {self.state}"
            )
        return self


class Artifact(BaseModel):
    """Artifact in the A2A protocol."""
    name: Optional[str] = None
    description: Optional[str] = None
    parts: List[Part]
    metadata: Optional[Dict[str, Any]] = None
    index: int = 0
    append: Optional[bool] = None
    lastChunk: Optional[bool] = None


class Task(BaseModel):
    """Task in the A2A protocol."""
    id: str
    sessionId: Optional[str] = None
    status: TaskStatus
    artifacts: Optional[List[Artifact]] = None
    history: Optional[List[Message]] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskStatusUpdateEvent(BaseModel):
    """Task status update event in the A2A protocol."""
    id: str
    status: TaskStatus
    final: bool = False
    metadata: Optional[Dict[str, Any]] = None


class TaskArtifactUpdateEvent(BaseModel):
    """Task artifact update event in the A2A protocol."""
    id: str
    artifact: Artifact
    metadata: Optional[Dict[str, Any]] = None


class AuthenticationInfo(BaseModel):
    """Authentication information in the A2A protocol."""
    model_config = ConfigDict(extra='allow')

    schemes: List[str]
    credentials: Optional[str] = None


class PushNotificationConfig(BaseModel):
    """Push notification configuration in the A2A protocol."""
    url: str
    token: Optional[str] = None
    authentication: Optional[AuthenticationInfo] = None


class TaskIdParams(BaseModel):
    """Task ID parameters in the A2A protocol."""
    id: str
    metadata: Optional[Dict[str, Any]] = None


class TaskQueryParams(TaskIdParams):
    """Task query parameters in the A2A protocol."""
    historyLength: Optional[int] = None


class TaskSendParams(BaseModel):
    """Task send parameters in the A2A protocol."""
    id: str
    sessionId: str = Field(default_factory=lambda: uuid4().hex)
    message: Message
    acceptedOutputModes: Optional[List[str]] = None
    pushNotification: Optional[PushNotificationConfig] = None
    historyLength: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskPushNotificationConfig(BaseModel):
    """Task push notification configuration in the A2A protocol."""
    id: str
    pushNotificationConfig: PushNotificationConfig



class JSONRPCMessage(BaseModel):
    """JSON-RPC message in the A2A protocol."""
    jsonrpc: Literal['2.0'] = '2.0'
    id: Optional[Union[int, str]] = Field(default_factory=lambda: uuid4().hex)


class JSONRPCRequest(JSONRPCMessage):
    """JSON-RPC request in the A2A protocol."""
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCError(BaseModel):
    """JSON-RPC error in the A2A protocol."""
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(JSONRPCMessage):
    """JSON-RPC response in the A2A protocol."""
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None


class SendTaskRequest(JSONRPCRequest):
    """Send task request in the A2A protocol."""
    method: Literal['tasks/send'] = 'tasks/send'
    params: TaskSendParams


class SendTaskResponse(JSONRPCResponse):
    """Send task response in the A2A protocol."""
    result: Optional[Task] = None


class SendTaskStreamingRequest(JSONRPCRequest):
    """Send task streaming request in the A2A protocol."""
    method: Literal['tasks/sendSubscribe'] = 'tasks/sendSubscribe'
    params: TaskSendParams


class SendTaskStreamingResponse(JSONRPCResponse):
    """Send task streaming response in the A2A protocol."""
    result: Optional[Union[TaskStatusUpdateEvent, TaskArtifactUpdateEvent]] = None


class GetTaskRequest(JSONRPCRequest):
    """Get task request in the A2A protocol."""
    method: Literal['tasks/get'] = 'tasks/get'
    params: TaskQueryParams


class GetTaskResponse(JSONRPCResponse):
    """Get task response in the A2A protocol."""
    result: Optional[Task] = None


class CancelTaskRequest(JSONRPCRequest):
    """Cancel task request in the A2A protocol."""
    method: Literal['tasks/cancel'] = 'tasks/cancel'
    params: TaskIdParams


class CancelTaskResponse(JSONRPCResponse):
    """Cancel task response in the A2A protocol."""
    result: Optional[Task] = None


class SetTaskPushNotificationRequest(JSONRPCRequest):
    """Set task push notification request in the A2A protocol."""
    method: Literal['tasks/pushNotification/set'] = 'tasks/pushNotification/set'
    params: TaskPushNotificationConfig


class SetTaskPushNotificationResponse(JSONRPCResponse):
    """Set task push notification response in the A2A protocol."""
    result: Optional[TaskPushNotificationConfig] = None


class GetTaskPushNotificationRequest(JSONRPCRequest):
    """Get task push notification request in the A2A protocol."""
    method: Literal['tasks/pushNotification/get'] = 'tasks/pushNotification/get'
    params: TaskIdParams


class GetTaskPushNotificationResponse(JSONRPCResponse):
    """Get task push notification response in the A2A protocol."""
    result: Optional[TaskPushNotificationConfig] = None


class TaskResubscriptionRequest(JSONRPCRequest):
    """Task resubscription request in the A2A protocol."""
    method: Literal['tasks/resubscribe'] = 'tasks/resubscribe'
    params: TaskIdParams


A2ARequest = TypeAdapter(
    Annotated[
        Union[
            SendTaskRequest,
            GetTaskRequest,
            CancelTaskRequest,
            SetTaskPushNotificationRequest,
            GetTaskPushNotificationRequest,
            TaskResubscriptionRequest,
            SendTaskStreamingRequest,
        ],
        Field(discriminator='method'),
    ]
)


class JSONParseError(JSONRPCError):
    """JSON parse error in the A2A protocol."""
    code: int = -32700
    message: str = 'Invalid JSON payload'
    data: Optional[Any] = None


class InvalidRequestError(JSONRPCError):
    """Invalid request error in the A2A protocol."""
    code: int = -32600
    message: str = 'Request payload validation error'
    data: Optional[Any] = None


class MethodNotFoundError(JSONRPCError):
    """Method not found error in the A2A protocol."""
    code: int = -32601
    message: str = 'Method not found'
    data: None = None


class InvalidParamsError(JSONRPCError):
    """Invalid parameters error in the A2A protocol."""
    code: int = -32602
    message: str = 'Invalid parameters'
    data: Optional[Any] = None


class InternalError(JSONRPCError):
    """Internal error in the A2A protocol."""
    code: int = -32603
    message: str = 'Internal error'
    data: Optional[Any] = None


class TaskNotFoundError(JSONRPCError):
    """Task not found error in the A2A protocol."""
    code: int = -32001
    message: str = 'Task not found'
    data: None = None


class TaskNotCancelableError(JSONRPCError):
    """Task not cancelable error in the A2A protocol."""
    code: int = -32002
    message: str = 'Task cannot be canceled'
    data: None = None


class PushNotificationNotSupportedError(JSONRPCError):
    """Push notification not supported error in the A2A protocol."""
    code: int = -32003
    message: str = 'Push Notification is not supported'
    data: None = None


class UnsupportedOperationError(JSONRPCError):
    """Unsupported operation error in the A2A protocol."""
    code: int = -32004
    message: str = 'This operation is not supported'
    data: None = None


class ContentTypeNotSupportedError(JSONRPCError):
    """Content type not supported error in the A2A protocol."""
    code: int = -32005
    message: str = 'Incompatible content types'
    data: None = None


class AgentProvider(BaseModel):
    """Agent provider in the A2A protocol."""
    organization: str
    url: Optional[str] = None


class AgentCapabilities(BaseModel):
    """Agent capabilities in the A2A protocol."""
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentAuthentication(BaseModel):
    """Agent authentication in the A2A protocol."""
    schemes: List[str]
    credentials: Optional[str] = None


class AgentSkill(BaseModel):
    """Agent skill in the A2A protocol."""
    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    inputModes: Optional[List[str]] = None
    outputModes: Optional[List[str]] = None


class AgentCard(BaseModel):
    """Agent card in the A2A protocol."""
    name: str
    description: Optional[str] = None
    url: str
    provider: Optional[AgentProvider] = None
    version: str
    documentationUrl: Optional[str] = None
    capabilities: AgentCapabilities
    authentication: Optional[AgentAuthentication] = None
    defaultInputModes: List[str] = ['text']
    defaultOutputModes: List[str] = ['text']
    skills: List[AgentSkill]


class A2AClientError(Exception):
    """Base exception for A2A client errors."""
    pass


class A2AClientHTTPError(A2AClientError):
    """HTTP error in the A2A client."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f'HTTP Error {status_code}: {message}')


class A2AClientJSONError(A2AClientError):
    """JSON error in the A2A client."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(f'JSON Error: {message}')


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""
    pass
