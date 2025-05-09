"""
A2A protocol client for CrewAI.

This module implements the client for the A2A protocol in CrewAI.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Union, cast

import aiohttp
from pydantic import ValidationError

if TYPE_CHECKING:
    from crewai.a2a.config import A2AConfig

from crewai.types.a2a import (
    A2AClientError,
    A2AClientHTTPError,
    A2AClientJSONError,
    Artifact,
    CancelTaskRequest,
    CancelTaskResponse,
    GetTaskPushNotificationRequest,
    GetTaskPushNotificationResponse,
    GetTaskRequest,
    GetTaskResponse,
    JSONRPCError,
    JSONRPCRequest,
    JSONRPCResponse,
    Message,
    MissingAPIKeyError,
    PushNotificationConfig,
    SendTaskRequest,
    SendTaskResponse,
    SendTaskStreamingRequest,
    SetTaskPushNotificationRequest,
    SetTaskPushNotificationResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskSendParams,
    TaskState,
    TaskStatusUpdateEvent,
)


class A2AClient:
    """A2A protocol client implementation."""

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        config: Optional["A2AConfig"] = None,
    ):
        """Initialize the A2A client.

        Args:
            base_url: The base URL of the A2A server.
            api_key: The API key to use for authentication.
            timeout: The timeout for HTTP requests in seconds.
            config: The A2A configuration. If provided, other parameters are ignored.
        """
        if config:
            from crewai.a2a.config import A2AConfig
            self.config = config
        else:
            from crewai.a2a.config import A2AConfig
            self.config = A2AConfig()
            if api_key:
                self.config.api_key = api_key
            if timeout:
                self.config.client_timeout = timeout
                
        self.base_url = base_url.rstrip("/")
        self.api_key = self.config.api_key or os.environ.get("A2A_API_KEY")
        self.timeout = self.config.client_timeout
        self.logger = logging.getLogger(__name__)

    async def send_task(
        self,
        task_id: str,
        message: Message,
        session_id: Optional[str] = None,
        accepted_output_modes: Optional[List[str]] = None,
        push_notification: Optional[PushNotificationConfig] = None,
        history_length: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Task:
        """Send a task to the A2A server.

        Args:
            task_id: The ID of the task.
            message: The message to send.
            session_id: The session ID.
            accepted_output_modes: The accepted output modes.
            push_notification: The push notification configuration.
            history_length: The number of messages to include in the history.
            metadata: Additional metadata.

        Returns:
            The created task.

        Raises:
            MissingAPIKeyError: If no API key is provided.
            A2AClientHTTPError: If there is an HTTP error.
            A2AClientJSONError: If there is an error parsing the JSON response.
            A2AClientError: If there is any other error sending the task.
        """
        params = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=message,
            acceptedOutputModes=accepted_output_modes,
            pushNotification=push_notification,
            historyLength=history_length,
            metadata=metadata,
        )

        request = SendTaskRequest(params=params)
        
        try:
            response = await self._send_jsonrpc_request(request)
            
            if response.error:
                raise A2AClientError(f"Error sending task: {response.error.message}")

            if not response.result:
                raise A2AClientError("No result returned from send task request")

            if isinstance(response.result, dict):
                return Task.model_validate(response.result)
            return cast(Task, response.result)
        except asyncio.TimeoutError as e:
            raise A2AClientError(f"Task request timed out: {e}")
        except aiohttp.ClientError as e:
            if isinstance(e, aiohttp.ClientResponseError):
                raise A2AClientHTTPError(e.status, str(e))
            else:
                raise A2AClientError(f"Client error: {e}")

    async def send_task_streaming(
        self,
        task_id: str,
        message: Message,
        session_id: Optional[str] = None,
        accepted_output_modes: Optional[List[str]] = None,
        push_notification: Optional[PushNotificationConfig] = None,
        history_length: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> asyncio.Queue:
        """Send a task to the A2A server and subscribe to updates.

        Args:
            task_id: The ID of the task.
            message: The message to send.
            session_id: The session ID.
            accepted_output_modes: The accepted output modes.
            push_notification: The push notification configuration.
            history_length: The number of messages to include in the history.
            metadata: Additional metadata.

        Returns:
            A queue that will receive task updates.

        Raises:
            A2AClientError: If there is an error sending the task.
        """
        params = TaskSendParams(
            id=task_id,
            sessionId=session_id,
            message=message,
            acceptedOutputModes=accepted_output_modes,
            pushNotification=push_notification,
            historyLength=history_length,
            metadata=metadata,
        )

        queue: asyncio.Queue = asyncio.Queue()

        asyncio.create_task(
            self._handle_streaming_response(
                f"{self.base_url}/v1/tasks/sendSubscribe", params, queue
            )
        )

        return queue

    async def get_task(
        self, task_id: str, history_length: Optional[int] = None
    ) -> Task:
        """Get a task from the A2A server.

        Args:
            task_id: The ID of the task.
            history_length: The number of messages to include in the history.

        Returns:
            The task.

        Raises:
            A2AClientError: If there is an error getting the task.
        """
        params = TaskQueryParams(id=task_id, historyLength=history_length)
        request = GetTaskRequest(params=params)
        response = await self._send_jsonrpc_request(request)

        if response.error:
            raise A2AClientError(f"Error getting task: {response.error.message}")

        if not response.result:
            raise A2AClientError("No result returned from get task request")

        return cast(Task, response.result)

    async def cancel_task(self, task_id: str) -> Task:
        """Cancel a task on the A2A server.

        Args:
            task_id: The ID of the task.

        Returns:
            The canceled task.

        Raises:
            A2AClientError: If there is an error canceling the task.
        """
        params = TaskIdParams(id=task_id)
        request = CancelTaskRequest(params=params)
        response = await self._send_jsonrpc_request(request)

        if response.error:
            raise A2AClientError(f"Error canceling task: {response.error.message}")

        if not response.result:
            raise A2AClientError("No result returned from cancel task request")

        return cast(Task, response.result)

    async def set_push_notification(
        self, task_id: str, config: PushNotificationConfig
    ) -> PushNotificationConfig:
        """Set push notification for a task.

        Args:
            task_id: The ID of the task.
            config: The push notification configuration.

        Returns:
            The push notification configuration.

        Raises:
            A2AClientError: If there is an error setting the push notification.
        """
        params = TaskPushNotificationConfig(id=task_id, pushNotificationConfig=config)
        request = SetTaskPushNotificationRequest(params=params)
        response = await self._send_jsonrpc_request(request)

        if response.error:
            raise A2AClientError(
                f"Error setting push notification: {response.error.message}"
            )

        if not response.result:
            raise A2AClientError(
                "No result returned from set push notification request"
            )

        return cast(TaskPushNotificationConfig, response.result).pushNotificationConfig

    async def get_push_notification(
        self, task_id: str
    ) -> Optional[PushNotificationConfig]:
        """Get push notification for a task.

        Args:
            task_id: The ID of the task.

        Returns:
            The push notification configuration, or None if not set.

        Raises:
            A2AClientError: If there is an error getting the push notification.
        """
        params = TaskIdParams(id=task_id)
        request = GetTaskPushNotificationRequest(params=params)
        response = await self._send_jsonrpc_request(request)

        if response.error:
            raise A2AClientError(
                f"Error getting push notification: {response.error.message}"
            )

        if not response.result:
            return None

        return cast(TaskPushNotificationConfig, response.result).pushNotificationConfig

    async def _send_jsonrpc_request(
        self, request: JSONRPCRequest
    ) -> JSONRPCResponse:
        """Send a JSON-RPC request to the A2A server.

        Args:
            request: The JSON-RPC request.

        Returns:
            The JSON-RPC response.

        Raises:
            A2AClientError: If there is an error sending the request.
        """
        if not self.api_key:
            raise MissingAPIKeyError(
                "API key is required. Set it in the constructor or as the A2A_API_KEY environment variable."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/jsonrpc",
                    headers=headers,
                    json=request.model_dump(),
                    timeout=self.timeout,
                ) as response:
                    if response.status != 200:
                        raise A2AClientHTTPError(
                            response.status, await response.text()
                        )

                    try:
                        data = await response.json()
                    except json.JSONDecodeError as e:
                        raise A2AClientJSONError(str(e))

                    try:
                        return JSONRPCResponse.model_validate(data)
                    except ValidationError as e:
                        raise A2AClientError(f"Invalid response: {e}")
        except aiohttp.ClientConnectorError as e:
            raise A2AClientHTTPError(status=0, message=f"Connection error: {e}")
        except aiohttp.ClientOSError as e:
            raise A2AClientHTTPError(status=0, message=f"OS error: {e}")
        except aiohttp.ServerDisconnectedError as e:
            raise A2AClientHTTPError(status=0, message=f"Server disconnected: {e}")
        except aiohttp.ClientResponseError as e:
            raise A2AClientHTTPError(e.status, str(e))
        except aiohttp.ClientError as e:
            raise A2AClientError(f"HTTP error: {e}")

    async def _handle_streaming_response(
        self,
        url: str,
        params: TaskSendParams,
        queue: asyncio.Queue,
    ) -> None:
        """Handle a streaming response from the A2A server.

        Args:
            url: The URL to send the request to.
            params: The task send parameters.
            queue: The queue to put events into.
        """
        if not self.api_key:
            await queue.put(
                Exception(
                    "API key is required. Set it in the constructor or as the A2A_API_KEY environment variable."
                )
            )
            return

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "text/event-stream",
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=headers,
                    json=params.model_dump(),
                    timeout=self.timeout,
                ) as response:
                    if response.status != 200:
                        await queue.put(
                            A2AClientHTTPError(response.status, await response.text())
                        )
                        return

                    buffer = ""
                    async for line in response.content:
                        line = line.decode("utf-8")
                        buffer += line

                        if buffer.endswith("\n\n"):
                            event_data = self._parse_sse_event(buffer)
                            buffer = ""

                            if event_data:
                                event_type = event_data.get("event")
                                data = event_data.get("data")

                                if event_type == "status":
                                    try:
                                        event = TaskStatusUpdateEvent.model_validate_json(data)
                                        await queue.put(event)

                                        if event.final:
                                            break
                                    except ValidationError as e:
                                        await queue.put(
                                            A2AClientError(f"Invalid status event: {e}")
                                        )
                                elif event_type == "artifact":
                                    try:
                                        event = TaskArtifactUpdateEvent.model_validate_json(data)
                                        await queue.put(event)
                                    except ValidationError as e:
                                        await queue.put(
                                            A2AClientError(f"Invalid artifact event: {e}")
                                        )
        except aiohttp.ClientConnectorError as e:
            await queue.put(A2AClientHTTPError(status=0, message=f"Connection error: {e}"))
        except aiohttp.ClientOSError as e:
            await queue.put(A2AClientHTTPError(status=0, message=f"OS error: {e}"))
        except aiohttp.ServerDisconnectedError as e:
            await queue.put(A2AClientHTTPError(status=0, message=f"Server disconnected: {e}"))
        except aiohttp.ClientResponseError as e:
            await queue.put(A2AClientHTTPError(e.status, str(e)))
        except aiohttp.ClientError as e:
            await queue.put(A2AClientError(f"HTTP error: {e}"))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await queue.put(A2AClientError(f"Error handling streaming response: {e}"))

    def _parse_sse_event(self, data: str) -> Dict[str, str]:
        """Parse an SSE event.

        Args:
            data: The SSE event data.

        Returns:
            A dictionary with the event type and data.
        """
        result = {}
        for line in data.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("event:"):
                result["event"] = line[6:].strip()
            elif line.startswith("data:"):
                result["data"] = line[5:].strip()

        return result
