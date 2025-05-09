"""
A2A protocol server for CrewAI.

This module implements the server for the A2A protocol in CrewAI.
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Type, TYPE_CHECKING, Union

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

if TYPE_CHECKING:
    from crewai.a2a.config import A2AConfig

from crewai.a2a.task_manager import InMemoryTaskManager, TaskManager
from crewai.types.a2a import (
    A2ARequest,
    CancelTaskRequest,
    CancelTaskResponse,
    ContentTypeNotSupportedError,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    GetTaskRequest,
    GetTaskResponse,
    InternalError,
    InvalidParamsError,
    InvalidRequestError,
    JSONParseError,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    MethodNotFoundError,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SendTaskStreamingResponse,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskNotCancelableError,
    TaskNotFoundError,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskSendParams,
    TaskState,
    TaskStatusUpdateEvent,
    UnsupportedOperationError,
)


class A2AServer:
    """A2A protocol server implementation."""

    def __init__(
        self,
        task_manager: Optional[TaskManager] = None,
        enable_cors: Optional[bool] = None,
        cors_origins: Optional[List[str]] = None,
        config: Optional["A2AConfig"] = None,
    ):
        """Initialize the A2A server.

        Args:
            task_manager: The task manager to use. If None, an InMemoryTaskManager will be created.
            enable_cors: Whether to enable CORS. If None, uses config value.
            cors_origins: The CORS origins to allow. If None, uses config value.
            config: The A2A configuration. If provided, other parameters are ignored.
        """
        from crewai.a2a.config import A2AConfig
        self.config = config or A2AConfig.from_env()
        
        enable_cors = enable_cors if enable_cors is not None else self.config.enable_cors
        cors_origins = cors_origins or self.config.cors_origins
        
        self.app = FastAPI(
            title="A2A Protocol Server",
            description="""
            A2A (Agent-to-Agent) protocol server for CrewAI.
            
            This server implements Google's A2A protocol specification, enabling interoperability
            between different agent systems. It provides endpoints for task creation, retrieval,
            cancellation, and streaming updates.
            """,
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_tags=[
                {
                    "name": "tasks",
                    "description": "Operations for managing A2A tasks",
                },
                {
                    "name": "jsonrpc",
                    "description": "JSON-RPC interface for the A2A protocol",
                },
            ],
        )
        self.task_manager = task_manager or InMemoryTaskManager()
        self.logger = logging.getLogger(__name__)

        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=cors_origins or ["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        @self.app.post(
            "/v1/jsonrpc",
            summary="Handle JSON-RPC requests",
            description="""
            Process JSON-RPC requests for the A2A protocol.
            
            This endpoint handles all JSON-RPC requests for the A2A protocol, including:
            - SendTask: Create a new task
            - GetTask: Retrieve a task by ID
            - CancelTask: Cancel a running task
            - SetTaskPushNotification: Configure push notifications for a task
            - GetTaskPushNotification: Retrieve push notification configuration for a task
            """,
            response_model=JSONRPCResponse,
            responses={
                200: {"description": "Successful response with result or error"},
                400: {"description": "Invalid request format or parameters"},
                500: {"description": "Internal server error during processing"},
            },
            tags=["jsonrpc"],
        )
        async def handle_jsonrpc(request: Request):
            return await self.handle_jsonrpc(request)
            
        @self.app.post(
            "/v1/tasks/send",
            summary="Send a task to an agent",
            description="""
            Create a new task and send it to an agent for execution.
            
            This endpoint allows clients to send tasks to agents for processing.
            The task is created with the provided parameters and immediately
            transitions to the WORKING state. The response includes the created
            task with its current status.
            """,
            response_model=Task,
            responses={
                200: {"description": "Task created successfully and processing started"},
                400: {"description": "Invalid request format or parameters"},
                500: {"description": "Internal server error during task creation or processing"},
            },
            tags=["tasks"],
        )
        async def handle_send_task(request: Request):
            return await self.handle_send_task(request)
            
        @self.app.post(
            "/v1/tasks/sendSubscribe",
            summary="Send a task and subscribe to updates",
            description="""
            Create a new task and subscribe to status updates via Server-Sent Events (SSE).
            
            This endpoint allows clients to send tasks to agents and receive real-time
            updates as the task progresses. The response is a streaming SSE connection
            that provides status updates and artifact notifications until the task
            reaches a terminal state (COMPLETED, FAILED, CANCELED, or EXPIRED).
            """,
            responses={
                200: {
                    "description": "Streaming response with task updates",
                    "content": {
                        "text/event-stream": {
                            "schema": {"type": "string"},
                            "example": 'event: status\ndata: {"task_id": "123", "status": {"state": "WORKING"}}\n\n',
                        }
                    },
                },
                400: {"description": "Invalid request format or parameters"},
                500: {"description": "Internal server error during task creation or processing"},
            },
            tags=["tasks"],
        )
        async def handle_send_task_subscribe(request: Request):
            return await self.handle_send_task_subscribe(request)
            
        @self.app.post(
            "/v1/tasks/{task_id}/cancel",
            summary="Cancel a task",
            description="""
            Cancel a running task by ID.
            
            This endpoint allows clients to cancel a task that is currently in progress.
            The task must be in a non-terminal state (PENDING, WORKING) to be canceled.
            Once canceled, the task transitions to the CANCELED state and cannot be
            resumed. The response includes the updated task with its current status.
            """,
            response_model=Task,
            responses={
                200: {"description": "Task canceled successfully and status updated to CANCELED"},
                404: {"description": "Task not found or already expired"},
                409: {"description": "Task cannot be canceled (already in terminal state)"},
                500: {"description": "Internal server error during task cancellation"},
            },
            tags=["tasks"],
        )
        async def handle_cancel_task(task_id: str, request: Request):
            return await self.handle_cancel_task(task_id, request)
            
        @self.app.get(
            "/v1/tasks/{task_id}",
            summary="Get task details",
            description="""
            Retrieve details of a task by ID.
            
            This endpoint allows clients to retrieve the current state and details of a task.
            The response includes the task's status, history, and any associated metadata.
            Clients can specify the history_length parameter to limit the number of messages
            included in the response.
            """,
            response_model=Task,
            responses={
                200: {"description": "Task details retrieved successfully with current status"},
                404: {"description": "Task not found or expired"},
                500: {"description": "Internal server error during task retrieval"},
            },
            tags=["tasks"],
        )
        async def handle_get_task(task_id: str, request: Request):
            return await self.handle_get_task(task_id, request)

    async def handle_jsonrpc(self, request: Request) -> JSONResponse:
        """Handle JSON-RPC requests.

        Args:
            request: The FastAPI request.

        Returns:
            A JSON response.
        """
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return JSONResponse(
                content=JSONRPCResponse(
                    id=None, error=JSONParseError()
                ).model_dump(),
                status_code=400,
            )

        try:
            if isinstance(body, list):
                responses = []
                for req_data in body:
                    response = await self._process_jsonrpc_request(req_data)
                    responses.append(response.model_dump())
                return JSONResponse(content=responses)
            else:
                response = await self._process_jsonrpc_request(body)
                return JSONResponse(content=response.model_dump())
        except Exception as e:
            self.logger.exception("Error processing JSON-RPC request")
            return JSONResponse(
                content=JSONRPCResponse(
                    id=body.get("id") if isinstance(body, dict) else None,
                    error=InternalError(message="Internal server error"),
                ).model_dump(),
                status_code=500,
            )

    async def _process_jsonrpc_request(
        self, request_data: Dict[str, Any]
    ) -> JSONRPCResponse:
        """Process a JSON-RPC request.

        Args:
            request_data: The JSON-RPC request data.

        Returns:
            A JSON-RPC response.
        """
        if not isinstance(request_data, dict) or request_data.get("jsonrpc") != "2.0":
            return JSONRPCResponse(
                id=request_data.get("id") if isinstance(request_data, dict) else None,
                error=InvalidRequestError(),
            )

        request_id = request_data.get("id")
        method = request_data.get("method")

        if not method:
            return JSONRPCResponse(
                id=request_id,
                error=InvalidRequestError(message="Method is required"),
            )

        try:
            request = A2ARequest.validate_python(request_data)
        except ValidationError as e:
            return JSONRPCResponse(
                id=request_id,
                error=InvalidParamsError(data=str(e)),
            )

        try:
            if isinstance(request, SendTaskRequest):
                task = await self._handle_send_task(request.params)
                return SendTaskResponse(id=request_id, result=task)
            elif isinstance(request, GetTaskRequest):
                task = await self.task_manager.get_task(
                    request.params.id, request.params.historyLength
                )
                return GetTaskResponse(id=request_id, result=task)
            elif isinstance(request, CancelTaskRequest):
                task = await self.task_manager.cancel_task(request.params.id)
                return CancelTaskResponse(id=request_id, result=task)
            elif isinstance(request, SetTaskPushNotificationRequest):
                config = await self.task_manager.set_push_notification(
                    request.params.id, request.params.pushNotificationConfig
                )
                return SetTaskPushNotificationResponse(
                    id=request_id, result=TaskPushNotificationConfig(id=request.params.id, pushNotificationConfig=config)
                )
            elif isinstance(request, GetTaskPushNotificationRequest):
                config = await self.task_manager.get_push_notification(
                    request.params.id
                )
                if config:
                    return GetTaskPushNotificationResponse(
                        id=request_id, result=TaskPushNotificationConfig(id=request.params.id, pushNotificationConfig=config)
                    )
                else:
                    return GetTaskPushNotificationResponse(id=request_id, result=None)
            elif isinstance(request, SendTaskStreamingRequest):
                return JSONRPCResponse(
                    id=request_id,
                    error=UnsupportedOperationError(
                        message="Streaming requests should be sent to the streaming endpoint"
                    ),
                )
            else:
                return JSONRPCResponse(
                    id=request_id,
                    error=MethodNotFoundError(),
                )
        except KeyError:
            return JSONRPCResponse(
                id=request_id,
                error=TaskNotFoundError(),
            )
        except Exception as e:
            self.logger.exception(f"Error handling {method} request")
            return JSONRPCResponse(
                id=request_id,
                error=InternalError(message="Internal server error"),
            )

    async def handle_send_task(self, request: Request) -> JSONResponse:
        """Handle send task requests.

        Args:
            request: The FastAPI request.

        Returns:
            A JSON response.
        """
        try:
            body = await request.json()
            params = TaskSendParams.model_validate(body)
            task = await self._handle_send_task(params)
            return JSONResponse(content=task.model_dump())
        except ValidationError:
            return JSONResponse(
                content={"error": "Invalid request format or parameters"},
                status_code=400,
            )
        except Exception as e:
            self.logger.exception("Error handling send task request")
            return JSONResponse(
                content={"error": "Internal server error"},
                status_code=500,
            )

    async def _handle_send_task(self, params: TaskSendParams) -> Task:
        """Handle send task requests.

        Args:
            params: The task send parameters.

        Returns:
            The created task.
        """
        task = await self.task_manager.create_task(
            task_id=params.id,
            session_id=params.sessionId,
            message=params.message,
            metadata=params.metadata,
        )
        
        await self.task_manager.update_task_status(
            task_id=params.id,
            state=TaskState.WORKING,
        )
        
        return task

    async def handle_send_task_subscribe(self, request: Request) -> StreamingResponse:
        """Handle send task subscribe requests.

        Args:
            request: The FastAPI request.

        Returns:
            A streaming response.
        """
        try:
            body = await request.json()
            params = TaskSendParams.model_validate(body)
            
            task = await self._handle_send_task(params)
            
            queue = await self.task_manager.subscribe_to_task(params.id)
            
            return StreamingResponse(
                self._stream_task_updates(params.id, queue),
                media_type="text/event-stream",
            )
        except ValidationError:
            return JSONResponse(
                content={"error": "Invalid request format or parameters"},
                status_code=400,
            )
        except Exception as e:
            self.logger.exception("Error handling send task subscribe request")
            return JSONResponse(
                content={"error": "Internal server error"},
                status_code=500,
            )

    async def _stream_task_updates(
        self, task_id: str, queue: asyncio.Queue
    ) -> None:
        """Stream task updates.

        Args:
            task_id: The ID of the task.
            queue: The queue to receive updates from.

        Yields:
            SSE formatted events.
        """
        try:
            while True:
                event = await queue.get()
                
                if isinstance(event, TaskStatusUpdateEvent):
                    event_type = "status"
                elif isinstance(event, TaskArtifactUpdateEvent):
                    event_type = "artifact"
                else:
                    event_type = "unknown"
                
                data = json.dumps(event.model_dump())
                yield f"event: {event_type}\ndata: {data}\n\n"
                
                if isinstance(event, TaskStatusUpdateEvent) and event.final:
                    break
        finally:
            await self.task_manager.unsubscribe_from_task(task_id, queue)

    async def handle_get_task(self, task_id: str, request: Request) -> JSONResponse:
        """Handle get task requests.

        Args:
            task_id: The ID of the task.
            request: The FastAPI request.

        Returns:
            A JSON response.
        """
        try:
            history_length = request.query_params.get("historyLength")
            history_length = int(history_length) if history_length else None
            
            task = await self.task_manager.get_task(task_id, history_length)
            return JSONResponse(content=task.model_dump())
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        except Exception as e:
            self.logger.exception(f"Error handling get task request for {task_id}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def handle_cancel_task(self, task_id: str, request: Request) -> JSONResponse:
        """Handle cancel task requests.

        Args:
            task_id: The ID of the task.
            request: The FastAPI request.

        Returns:
            A JSON response.
        """
        try:
            task = await self.task_manager.cancel_task(task_id)
            return JSONResponse(content=task.model_dump())
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        except Exception as e:
            self.logger.exception(f"Error handling cancel task request for {task_id}")
            raise HTTPException(status_code=500, detail="Internal server error")
