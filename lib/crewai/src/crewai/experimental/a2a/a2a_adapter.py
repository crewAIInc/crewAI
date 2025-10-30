"""A2A Agent Adapter implementation for CrewAI.

This module provides the main adapter class for integrating A2A protocol-compliant
agents into CrewAI workflows.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Literal
import uuid

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import (
    GetTaskPushNotificationConfigParams,
    Message,
    Part,
    PushNotificationAuthenticationInfo,
    PushNotificationConfig,
    Role,
    TaskIdParams,
    TaskPushNotificationConfig,
    TaskQueryParams,
    TaskState,
    TextPart,
    TransportProtocol,
)
import httpx
from pydantic import Field, PrivateAttr

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.experimental.a2a.auth import AuthScheme, BearerTokenAuth
from crewai.experimental.a2a.exceptions import (
    A2AAuthenticationError,
    A2AConfigurationError,
    A2AConnectionError,
    A2AInputRequiredError,
    A2ATaskCanceledError,
    A2ATaskFailedError,
)
from crewai.tools.base_tool import BaseTool


if TYPE_CHECKING:
    from a2a.types import AgentCard


class A2AAgentAdapter(BaseAgent):
    """Adapter for A2A protocol-compliant agents.

    Integrates external A2A agents (ServiceNow, Bedrock, Glean, etc.) into CrewAI workflows.
    Uses the official a2a-sdk for protocol compliance and multi-transport support.

    The adapter handles:
    - AgentCard discovery and validation
    - Message formatting and translation
    - Task lifecycle management (creation, cancellation, retrieval)
    - Streaming and polling execution modes with automatic selection
    - Async/sync bridging
    - Authentication via Bearer tokens

    Attributes:
        agent_card_url: URL to the A2A AgentCard (supports .well-known/agent-card.json).
        auth_token: Optional Bearer token for authentication.
        timeout: Request timeout in seconds (default: 120).
        preferred_transport: Preferred transport protocol (default: "JSONRPC").
            Supported: "JSONRPC", "GRPC", "HTTP+JSON".
        enable_streaming: Whether to enable streaming responses (default: True).

    Example:
        ```python
        from crewai import Agent, Task, Crew
        from crewai.experimental.a2a import A2AAgentAdapter

        servicenow_agent = A2AAgentAdapter(
            agent_card_url="https://servicenow.example.com/.well-known/agent-card.json",
            auth_token="your-token-here",
            role="ServiceNow Incident Manager",
            goal="Create and manage IT incidents",
            backstory="Expert at incident management with 10 years experience",
        )

        task = Task(
            description="Create a P1 incident for database outage",
            expected_output="Incident ticket number and details",
            agent=servicenow_agent,
        )

        crew = Crew(agents=[servicenow_agent], tasks=[task])
        result = crew.kickoff()
        ```

    Note:
        Requires a2a-sdk to be installed:
        ```bash
        uv add 'crewai[a2a]'
        ```
    """

    agent_card_url: str = Field(
        description="URL to the A2A AgentCard (supports .well-known/agent-card.json)"
    )
    auth_token: str | None = Field(
        default=None,
        description="Bearer token for authentication (deprecated: use auth_scheme)",
    )
    auth_scheme: AuthScheme | None = Field(
        default=None,
        description="Authentication scheme (Bearer, OAuth2, API Key, HTTP Basic/Digest)",
    )
    timeout: int = Field(default=120, description="Request timeout in seconds")
    preferred_transport: Literal["JSONRPC", "GRPC", "HTTP+JSON", "HTTP_JSON"] = Field(
        default="JSONRPC",
        description="Preferred transport protocol (JSONRPC, GRPC, HTTP+JSON)",
    )
    enable_streaming: bool = Field(
        default=True, description="Whether to enable streaming responses"
    )
    adapted_agent: bool = Field(default=True, init=False)
    function_calling_llm: Any = Field(
        default=None, description="Not used for A2A agents"
    )
    step_callback: Any = Field(default=None, description="Not used for A2A agents")

    _agent_card: AgentCard | None = PrivateAttr(default=None)
    _a2a_sdk_available: bool = PrivateAttr(default=False)
    _headers: dict[str, str] = PrivateAttr(default_factory=dict)
    _transport_protocol: TransportProtocol | None = PrivateAttr(default=None)
    _base_url: str = PrivateAttr(default="")
    _current_task_id: str | None = PrivateAttr(default=None)

    def __init__(self, **data):
        """Initialize A2A adapter.

        Raises:
            ImportError: If a2a-sdk is not installed.
        """
        super().__init__(**data)

        try:
            import a2a  # noqa: F401

            self._a2a_sdk_available = True
        except ImportError as e:
            msg = (
                "A2A SDK not installed. Install with: uv add 'crewai[a2a]' "
                "or uv add 'a2a-sdk>=0.1.0'"
            )
            raise ImportError(msg) from e

    def create_agent_executor(self, tools: list[BaseTool] | None = None) -> None:
        """Initialize the A2A agent configuration and fetch AgentCard.

        This method:
        1. Sets up authentication headers
        2. Maps transport protocol
        3. Discovers the AgentCard from the provided URL
        4. Stores configuration for later use

        Args:
            tools: Optional list of tools (not used for A2A agents as they define their own skills).

        Raises:
            A2AConfigurationError: If a2a-sdk is not installed.
            A2AConnectionError: If AgentCard discovery or client initialization fails.
        """
        if not self._a2a_sdk_available:
            msg = "A2A SDK not available. Install with: pip install 'crewai[a2a]'"
            raise ImportError(msg)

        # Handle backward compatibility: auth_token -> auth_scheme
        if self.auth_token and not self.auth_scheme:
            self.auth_scheme = BearerTokenAuth(token=self.auth_token)

        transport_map = {
            "JSONRPC": TransportProtocol.jsonrpc,
            "GRPC": TransportProtocol.grpc,
            "HTTP+JSON": TransportProtocol.http_json,
            "HTTP_JSON": TransportProtocol.http_json,
        }
        self._transport_protocol = transport_map.get(
            self.preferred_transport.upper(), TransportProtocol.http_json
        )

        agent_card_url = self.agent_card_url

        if "/.well-known/" in agent_card_url:
            base_url, path_part = agent_card_url.rsplit("/.well-known/", 1)
            agent_card_path = f"/.well-known/{path_part}"
        else:
            base_url = agent_card_url
            agent_card_path = "/.well-known/agent-card.json"

        self._base_url = base_url

        async def _fetch_agent_card():
            async with httpx.AsyncClient(
                timeout=self.timeout, headers=self._headers
            ) as httpx_client:
                # Configure authentication on the client
                if self.auth_scheme:
                    self.auth_scheme.configure_client(httpx_client)
                    # Apply auth to headers
                    self._headers = await self.auth_scheme.apply_auth(
                        httpx_client, self._headers
                    )

                resolver = A2ACardResolver(
                    httpx_client=httpx_client,
                    base_url=base_url,
                    agent_card_path=agent_card_path,
                )
                return await resolver.get_agent_card()

        self._agent_card = asyncio.run(_fetch_agent_card())

        self._logger.log(
            "info",
            f"A2A agent initialized: {self._agent_card.name} v{self._agent_card.version}",
        )
        self._logger.log(
            "info",
            f"Skills available: {len(self._agent_card.skills)} | "
            f"Streaming: {self._agent_card.capabilities.streaming}",
        )

        self._check_io_mode_compatibility()
        self._check_state_transition_history()

    def execute_task(
        self,
        task: Any,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        """Execute a CrewAI task via A2A protocol.

        Converts the CrewAI task to an A2A message, sends it to the agent,
        and aggregates the response(s) into a string result.

        The execution flow:
        1. Build A2A message from task description and context
        2. Send message to A2A agent (streaming or blocking)
        3. Process responses/events
        4. Extract final result from task history or artifacts
        5. Handle error states (input_required, failed, etc.)

        Args:
            task: CrewAI Task object containing description and expected output.
            context: Optional context string from previous tasks.
            tools: Optional tools (not used - A2A agents define their own skills).

        Returns:
            String result from the A2A agent execution.

        Raises:
            A2ATaskFailedError: If the A2A agent task fails or is rejected.
            A2AInputRequiredError: If the A2A agent requires additional input.
            A2AAuthenticationError: If the A2A agent requires authentication.
            A2ATaskCanceledError: If the A2A task is canceled.
            A2AConnectionError: If connection to the A2A agent fails.
        """
        if not self._agent_card:
            self.create_agent_executor(tools)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._execute_async(task, context, tools))
        finally:
            loop.close()

    async def _execute_async(
        self,
        task: Any,
        context: str | None,
        tools: list[BaseTool] | None,
    ) -> str:
        """Async implementation of task execution via A2A protocol.

        Automatically selects between streaming and polling modes based on:
        - Agent capabilities (AgentCard.capabilities.streaming)
        - Configuration (enable_streaming)
        - Fallback on streaming errors

        Args:
            task: CrewAI Task object.
            context: Optional context from previous tasks.
            tools: Optional tools (not used).

        Returns:
            String result from A2A agent.

        Raises:
            A2ATaskFailedError: If the A2A agent task fails or is rejected.
            A2AInputRequiredError: If the A2A agent requires additional input.
            A2AAuthenticationError: If the A2A agent requires authentication.
            A2ATaskCanceledError: If the A2A task is canceled.
            A2AConnectionError: If connection to the A2A agent fails.
        """
        streaming_supported = (
            self._agent_card.capabilities.streaming
            if self._agent_card
            and self._agent_card.capabilities
            and self._agent_card.capabilities.streaming is not None
            else True
        )

        use_streaming = self.enable_streaming and streaming_supported

        if not streaming_supported:
            self._logger.log(
                "info", "Agent does not support streaming, using polling mode"
            )

        if use_streaming:
            try:
                return await self._execute_streaming(task, context)
            except Exception as e:
                self._logger.log(
                    "warning", f"Streaming failed ({e}), falling back to polling mode"
                )
                return await self._execute_polling(task, context)

        return await self._execute_polling(task, context)

    def _extract_artifacts_with_metadata(self, artifacts: list[Any]) -> str:
        """Extract artifacts with full metadata preservation.

        Args:
            artifacts: List of A2A Artifact objects.

        Returns:
            JSON-formatted string containing artifact data with metadata.
        """
        artifacts_data = []

        for artifact in artifacts:
            artifact_content: dict[str, Any] = {
                "id": artifact.artifact_id,
                "name": artifact.name,
                "description": artifact.description,
                "parts": [],
            }

            if artifact.metadata:
                artifact_content["metadata"] = artifact.metadata

            for part in artifact.parts:
                if part.root.kind == "text":
                    artifact_content["parts"].append(
                        {
                            "type": "text",
                            "content": part.root.text,
                        }
                    )
                elif part.root.kind == "file":
                    part_data: dict[str, str] = {
                        "type": "file",
                        "uri": part.root.file.uri,
                    }
                    if part.root.file.mime_type:
                        part_data["media_type"] = part.root.file.mime_type
                    artifact_content["parts"].append(part_data)
                elif part.root.kind == "data":
                    artifact_content["parts"].append(
                        {
                            "type": "data",
                            "data": part.root.data,
                        }
                    )

            artifacts_data.append(artifact_content)

        return f"\n\nArtifacts:\n{json.dumps(artifacts_data, indent=2)}"

    async def _execute_streaming(
        self,
        task: Any,
        context: str | None,
    ) -> str:
        """Execute task using streaming mode with automatic reconnection.

        This method implements automatic reconnection on network failures using
        the A2A protocol's resubscribe functionality. If the connection drops
        mid-stream, it will attempt to reconnect up to 3 times with exponential backoff.

        Args:
            task: CrewAI Task object.
            context: Optional context from previous tasks.

        Returns:
            String result from A2A agent.

        Raises:
            A2ATaskFailedError: If the A2A agent task fails or is rejected.
            A2AInputRequiredError: If the A2A agent requires additional input.
            A2AAuthenticationError: If the A2A agent requires authentication.
            A2ATaskCanceledError: If the A2A task is canceled.
            A2AConnectionError: If connection to the A2A agent fails after all retries.
        """
        max_retries = 3
        saved_task_id: str | None = None

        for attempt in range(max_retries):
            try:
                return await self._execute_streaming_attempt(
                    task, context, saved_task_id
                )
            except (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ) as e:
                if saved_task_id is None:
                    saved_task_id = self._current_task_id

                if attempt < max_retries - 1 and saved_task_id:
                    backoff_time = 2**attempt
                    self._logger.log(
                        "warning",
                        f"Connection lost ({type(e).__name__}), retrying in {backoff_time}s "
                        f"(attempt {attempt + 1}/{max_retries})...",
                    )
                    await asyncio.sleep(backoff_time)
                    continue

                error_msg = f"Connection failed after {max_retries} attempts: {e}"
                self._logger.log("error", error_msg)
                raise A2AConnectionError(error_msg) from e
            except (
                A2ATaskFailedError,
                A2AInputRequiredError,
                A2AAuthenticationError,
                A2ATaskCanceledError,
            ):
                raise

        msg = f"Streaming execution failed after {max_retries} attempts"
        raise A2AConnectionError(msg)

    async def _execute_streaming_attempt(
        self,
        task: Any,
        context: str | None,
        resubscribe_task_id: str | None = None,
    ) -> str:
        """Single attempt at streaming execution, with optional resubscription.

        Args:
            task: CrewAI Task object.
            context: Optional context from previous tasks.
            resubscribe_task_id: Optional task ID to resubscribe to (for reconnection).

        Returns:
            String result from A2A agent.

        Raises:
            A2ATaskFailedError: If the A2A agent task fails or is rejected.
            A2AInputRequiredError: If the A2A agent requires additional input.
            A2AAuthenticationError: If the A2A agent requires authentication.
            A2ATaskCanceledError: If the A2A task is canceled.
            httpx.TimeoutException, httpx.NetworkError: If connection fails.
        """
        content_parts = []

        if context:
            content_parts.append(f"Context:\n{context}\n\n")

        content_parts.append(f"Task: {task.description}\n")
        content_parts.append(f"Expected Output: {task.expected_output}")

        message_text = "".join(content_parts)

        if not self._agent_card:
            msg = "Agent card not initialized"
            raise A2AConfigurationError(msg)

        message = Message(
            role=Role.user,
            message_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=message_text))],
        )

        self._logger.log(
            "info", f"Sending task to A2A agent (streaming): {self._agent_card.name}"
        )

        if not self._transport_protocol:
            msg = "Transport protocol not configured"
            raise A2AConfigurationError(msg)

        async with httpx.AsyncClient(
            timeout=self.timeout, headers=self._headers
        ) as httpx_client:
            config = ClientConfig(
                httpx_client=httpx_client,
                supported_transports=[str(self._transport_protocol.value)],
                streaming=self.enable_streaming,
            )

            factory = ClientFactory(config)
            client = factory.create(self._agent_card)

            result_parts = []

            if resubscribe_task_id:
                self._logger.log(
                    "info", f"Resubscribing to task: {resubscribe_task_id}"
                )
                params = TaskIdParams(id=resubscribe_task_id)
                event_stream = client.resubscribe(params)
            else:
                # send_message returns Message | tuple, so we can't type-narrow event_stream
                event_stream = client.send_message(message)  # type: ignore[assignment]

            async for event in event_stream:
                if isinstance(event, Message):
                    self._logger.log("debug", "Received direct message response")
                    for part in event.parts:
                        if part.root.kind == "text":
                            result_parts.append(part.root.text)

                elif isinstance(event, tuple):
                    a2a_task, update = event

                    if a2a_task.id and not self._current_task_id:
                        self._current_task_id = a2a_task.id
                        self._logger.log(
                            "debug", f"Tracking task ID: {self._current_task_id}"
                        )

                    self._logger.log(
                        "debug",
                        f"Task state: {a2a_task.status.state} | "
                        f"Update: {type(update).__name__ if update else 'None'}",
                    )

                    if a2a_task.status.state == TaskState.completed:
                        if a2a_task.history:
                            for history_msg in reversed(a2a_task.history):
                                if history_msg.role == Role.agent:
                                    for part in history_msg.parts:
                                        if part.root.kind == "text":
                                            result_parts.append(part.root.text)
                                    break

                        if a2a_task.artifacts:
                            artifacts_text = self._extract_artifacts_with_metadata(
                                a2a_task.artifacts
                            )
                            result_parts.append(artifacts_text)

                        self._logger.log("info", "Task completed successfully")
                        break

                    if a2a_task.status.state in [
                        TaskState.failed,
                        TaskState.rejected,
                    ]:
                        error_msg = "Task failed without error message"
                        if a2a_task.status.message and a2a_task.status.message.parts:
                            first_part = a2a_task.status.message.parts[0]
                            if first_part.root.kind == "text":
                                error_msg = first_part.root.text
                        self._logger.log("error", f"Task failed: {error_msg}")
                        raise A2ATaskFailedError(error_msg)

                    if a2a_task.status.state == TaskState.input_required:
                        error_msg = "Additional input required"
                        if a2a_task.status.message and a2a_task.status.message.parts:
                            first_part = a2a_task.status.message.parts[0]
                            if first_part.root.kind == "text":
                                error_msg = first_part.root.text
                        self._logger.log("warning", f"Task requires input: {error_msg}")
                        raise A2AInputRequiredError(error_msg)

                    if a2a_task.status.state == TaskState.auth_required:
                        error_msg = "Authentication required to continue"
                        if a2a_task.status.message and a2a_task.status.message.parts:
                            first_part = a2a_task.status.message.parts[0]
                            if first_part.root.kind == "text":
                                error_msg = first_part.root.text
                        self._logger.log(
                            "error", f"Task requires authentication: {error_msg}"
                        )
                        raise A2AAuthenticationError(error_msg)

                    if a2a_task.status.state == TaskState.canceled:
                        error_msg = "Task was canceled"
                        if a2a_task.status.message and a2a_task.status.message.parts:
                            first_part = a2a_task.status.message.parts[0]
                            if first_part.root.kind == "text":
                                error_msg = first_part.root.text
                        self._logger.log("warning", f"Task canceled: {error_msg}")
                        raise A2ATaskCanceledError(error_msg)

                    if a2a_task.status.state == TaskState.unknown:
                        self._logger.log(
                            "warning",
                            "Task in unknown state, continuing to wait for state change...",
                        )

            result = (
                "\n".join(result_parts)
                if result_parts
                else "No response from A2A agent"
            )
            self._logger.log(
                "info", f"A2A execution complete. Result length: {len(result)} chars"
            )

            self._current_task_id = None

            return result

    async def _execute_polling(
        self,
        task: Any,
        context: str | None,
    ) -> str:
        """Execute task using polling mode.

        This method sends the initial message to create a task, then polls
        the task status using get_task() until completion.

        Args:
            task: CrewAI Task object.
            context: Optional context from previous tasks.

        Returns:
            String result from A2A agent.

        Raises:
            A2ATaskFailedError: If the A2A agent task fails or is rejected.
            A2AInputRequiredError: If the A2A agent requires additional input.
            A2AAuthenticationError: If the A2A agent requires authentication.
            A2ATaskCanceledError: If the A2A task is canceled.
            A2AConnectionError: If connection to the A2A agent fails.
        """
        content_parts = []

        if context:
            content_parts.append(f"Context:\n{context}\n\n")

        content_parts.append(f"Task: {task.description}\n")
        content_parts.append(f"Expected Output: {task.expected_output}")

        message_text = "".join(content_parts)

        message = Message(
            role=Role.user,
            message_id=str(uuid.uuid4()),
            parts=[Part(root=TextPart(text=message_text))],
        )

        if not self._agent_card:
            msg = "Agent card not initialized"
            raise A2AConfigurationError(msg)

        if not self._transport_protocol:
            msg = "Transport protocol not configured"
            raise A2AConfigurationError(msg)

        self._logger.log(
            "info", f"Sending task to A2A agent (polling): {self._agent_card.name}"
        )

        async with httpx.AsyncClient(
            timeout=self.timeout, headers=self._headers
        ) as httpx_client:
            config = ClientConfig(
                httpx_client=httpx_client,
                supported_transports=[str(self._transport_protocol.value)],
                streaming=False,
            )

            factory = ClientFactory(config)
            client = factory.create(self._agent_card)

            task_id = None

            async for event in client.send_message(message):
                if isinstance(event, tuple):
                    a2a_task, _ = event
                    if a2a_task.id:
                        task_id = a2a_task.id
                        self._current_task_id = task_id
                        self._logger.log("info", f"Task created with ID: {task_id}")
                        break

            if not task_id:
                msg = "Failed to obtain task ID from A2A agent"
                self._logger.log("error", msg)
                raise A2AConnectionError(msg)

            self._logger.log("debug", "Starting polling loop")
            poll_interval = 2
            max_polls = self.timeout // poll_interval
            poll_count = 0

            while poll_count < max_polls:
                poll_count += 1
                await asyncio.sleep(poll_interval)

                params = TaskQueryParams(id=task_id)
                a2a_task = await client.get_task(params)

                self._logger.log(
                    "debug", f"Poll {poll_count}: Task state = {a2a_task.status.state}"
                )

                if a2a_task.status.state == TaskState.completed:
                    self._logger.log("info", "Task completed successfully")

                    result_parts = []
                    if a2a_task.history:
                        for history_msg in reversed(a2a_task.history):
                            if history_msg.role == Role.agent:
                                for part in history_msg.parts:
                                    if part.root.kind == "text":
                                        result_parts.append(part.root.text)
                                break

                    if a2a_task.artifacts:
                        artifacts_text = self._extract_artifacts_with_metadata(
                            a2a_task.artifacts
                        )
                        result_parts.append(artifacts_text)

                    result = (
                        "\n".join(result_parts)
                        if result_parts
                        else "No response from A2A agent"
                    )
                    self._logger.log(
                        "info",
                        f"A2A execution complete. Result length: {len(result)} chars",
                    )

                    self._current_task_id = None

                    return result

                if a2a_task.status.state in [TaskState.failed, TaskState.rejected]:
                    error_msg = "Task failed without error message"
                    if a2a_task.status.message and a2a_task.status.message.parts:
                        first_part = a2a_task.status.message.parts[0]
                        if first_part.root.kind == "text":
                            error_msg = first_part.root.text
                    self._logger.log("error", f"Task failed: {error_msg}")
                    self._current_task_id = None
                    raise A2ATaskFailedError(error_msg)

                if a2a_task.status.state == TaskState.input_required:
                    error_msg = "Additional input required"
                    if a2a_task.status.message and a2a_task.status.message.parts:
                        first_part = a2a_task.status.message.parts[0]
                        if first_part.root.kind == "text":
                            error_msg = first_part.root.text
                    self._logger.log("warning", f"Task requires input: {error_msg}")
                    self._current_task_id = None
                    raise A2AInputRequiredError(error_msg)

                if a2a_task.status.state == TaskState.auth_required:
                    error_msg = "Authentication required to continue"
                    if a2a_task.status.message and a2a_task.status.message.parts:
                        first_part = a2a_task.status.message.parts[0]
                        if first_part.root.kind == "text":
                            error_msg = first_part.root.text
                    self._logger.log(
                        "error", f"Task requires authentication: {error_msg}"
                    )
                    self._current_task_id = None
                    raise A2AAuthenticationError(error_msg)

                if a2a_task.status.state == TaskState.canceled:
                    error_msg = "Task was canceled"
                    if a2a_task.status.message and a2a_task.status.message.parts:
                        first_part = a2a_task.status.message.parts[0]
                        if first_part.root.kind == "text":
                            error_msg = first_part.root.text
                    self._logger.log("warning", f"Task canceled: {error_msg}")
                    self._current_task_id = None
                    raise A2ATaskCanceledError(error_msg)

                if a2a_task.status.state == TaskState.unknown:
                    self._logger.log(
                        "warning",
                        f"Task in unknown state, continuing to poll (attempt {poll_count}/{max_polls})...",
                    )

            msg = f"Task polling timeout after {self.timeout} seconds"
            self._logger.log("error", msg)
            self._current_task_id = None
            raise A2AConnectionError(msg)

    def cancel_task(self, task_id: str | None = None) -> bool:
        """Cancel a running A2A task.

        Args:
            task_id: The ID of the task to cancel. If None, cancels the currently
                executing task (if any).

        Returns:
            True if task was successfully canceled, False otherwise.

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2AConfigurationError: If agent card is not initialized.

        Example:
            ```python
            adapter.cancel_task()
            adapter.cancel_task("task-123")
            ```
        """
        if not self._agent_card:
            msg = "Agent card not initialized. Call create_agent_executor() first."
            raise A2AConfigurationError(msg)

        cancel_id = task_id or self._current_task_id

        if not cancel_id:
            self._logger.log("warning", "No task ID to cancel")
            return False

        self._logger.log("info", f"Canceling task: {cancel_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._cancel_task_async(cancel_id))
        finally:
            loop.close()

    async def _cancel_task_async(self, task_id: str) -> bool:
        """Async implementation of task cancellation.

        Args:
            task_id: The ID of the task to cancel.

        Returns:
            True if cancellation was successful, False otherwise.

        Raises:
            A2AConnectionError: If connection to agent fails.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers=self._headers
            ) as httpx_client:
                if not self._agent_card:
                    msg = "Agent card not initialized"
                    raise A2AConfigurationError(msg)

                if not self._transport_protocol:
                    msg = "Transport protocol not configured"
                    raise A2AConfigurationError(msg)

                config = ClientConfig(
                    httpx_client=httpx_client,
                    supported_transports=[str(self._transport_protocol.value)],
                    streaming=self.enable_streaming,
                )

                factory = ClientFactory(config)
                client = factory.create(self._agent_card)

                params = TaskIdParams(id=task_id)
                await client.cancel_task(params)

                self._logger.log("info", f"Task {task_id} canceled successfully")

                if self._current_task_id == task_id:
                    self._current_task_id = None

                return True

        except Exception as e:
            self._logger.log("error", f"Failed to cancel task {task_id}: {e}")
            raise A2AConnectionError(f"Failed to cancel task: {e}") from e

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Retrieve the current status and details of an A2A task.

        This method allows polling for task status, which is useful for:
        - Checking task progress after disconnection
        - Retrieving final results without streaming
        - Monitoring long-running tasks

        Args:
            task_id: The ID of the task to retrieve.

        Returns:
            Dictionary containing task information with keys:
            - task_id: The task identifier
            - state: Current task state (e.g., "completed", "working", "failed")
            - result: Task result (if completed)
            - error: Error message (if failed)
            - history: Message history
            - artifacts: Task artifacts

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2AConfigurationError: If agent card is not initialized.

        Example:
            ```python
            task_info = adapter.get_task("task-123")
            print(f"State: {task_info['state']}")
            if task_info["state"] == "completed":
                print(f"Result: {task_info['result']}")
            ```
        """
        if not self._agent_card:
            msg = "Agent card not initialized. Call create_agent_executor() first."
            raise A2AConfigurationError(msg)

        self._logger.log("info", f"Retrieving task: {task_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_task_async(task_id))
        finally:
            loop.close()

    async def _get_task_async(self, task_id: str) -> dict[str, Any]:
        """Async implementation of task retrieval.

        Args:
            task_id: The ID of the task to retrieve.

        Returns:
            Dictionary with task information.

        Raises:
            A2AConnectionError: If connection to agent fails.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers=self._headers
            ) as httpx_client:
                if not self._agent_card:
                    msg = "Agent card not initialized"
                    raise A2AConfigurationError(msg)

                if not self._transport_protocol:
                    msg = "Transport protocol not configured"
                    raise A2AConfigurationError(msg)

                config = ClientConfig(
                    httpx_client=httpx_client,
                    supported_transports=[str(self._transport_protocol.value)],
                    streaming=self.enable_streaming,
                )

                factory = ClientFactory(config)
                client = factory.create(self._agent_card)

                params = TaskQueryParams(id=task_id)
                a2a_task = await client.get_task(params)

                task_info: dict[str, Any] = {
                    "task_id": a2a_task.id,
                    "state": str(a2a_task.status.state),
                    "result": None,
                    "error": None,
                    "history": [],
                    "artifacts": [],
                }

                if a2a_task.history:
                    for history_msg in reversed(a2a_task.history):
                        if history_msg.role == Role.agent:
                            text_parts = []
                            for part in history_msg.parts:
                                if part.root.kind == "text":
                                    text_parts.append(part.root.text)
                            if text_parts:
                                task_info["result"] = "\n".join(text_parts)
                                break

                    task_info["history"] = [
                        {
                            "role": str(history_msg.role),
                            "content": [
                                part.root.text
                                for part in history_msg.parts
                                if part.root.kind == "text"
                            ],
                        }
                        for history_msg in a2a_task.history
                    ]

                if a2a_task.artifacts:
                    artifact_list: list[dict[str, Any]] = []
                    for artifact in a2a_task.artifacts:
                        artifact_data = {
                            "id": artifact.artifact_id,
                            "name": artifact.name,
                            "description": artifact.description,
                            "parts": [
                                part.root.text
                                for part in artifact.parts
                                if part.root.kind == "text"
                            ],
                        }
                        artifact_list.append(artifact_data)
                    task_info["artifacts"] = artifact_list

                if a2a_task.status.message and a2a_task.status.message.parts:
                    first_part = a2a_task.status.message.parts[0]
                    if first_part.root.kind == "text":
                        task_info["error"] = first_part.root.text

                self._logger.log(
                    "info", f"Retrieved task {task_id}: state={task_info['state']}"
                )
                return task_info

        except Exception as e:
            self._logger.log("error", f"Failed to retrieve task {task_id}: {e}")
            raise A2AConnectionError(f"Failed to retrieve task: {e}") from e

    def set_task_callback(
        self,
        task_id: str,
        webhook_url: str,
        auth_token: str | None = None,
    ) -> dict[str, Any]:
        """Configure push notification webhook for a task.

        This method allows you to set up a webhook that the A2A agent will call
        when the task state changes, instead of requiring streaming or polling.

        Note: You must provide your own webhook server to receive notifications.
        This method only configures the A2A agent to send notifications to your URL.

        Args:
            task_id: The ID of the task to configure notifications for.
            webhook_url: The URL where the A2A agent should send notifications.
            auth_token: Optional Bearer token for authenticating webhook requests.

        Returns:
            Dictionary containing the push notification configuration with keys:
            - config_id: The configuration identifier
            - url: The webhook URL
            - task_id: The associated task ID

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2AConfigurationError: If agent card is not initialized.

        Example:
            ```python
            config = adapter.set_task_callback(
                task_id="task-123",
                webhook_url="https://myapp.com/webhooks/a2a",
                auth_token="my-webhook-secret",
            )
            print(f"Webhook configured: {config['config_id']}")
            ```
        """
        if not self._agent_card:
            msg = "Agent card not initialized. Call create_agent_executor() first."
            raise A2AConfigurationError(msg)

        self._logger.log("info", f"Configuring webhook for task: {task_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._set_task_callback_async(task_id, webhook_url, auth_token)
            )
        finally:
            loop.close()

    async def _set_task_callback_async(
        self,
        task_id: str,
        webhook_url: str,
        auth_token: str | None,
    ) -> dict[str, Any]:
        """Async implementation of webhook configuration.

        Args:
            task_id: The ID of the task.
            webhook_url: The webhook URL.
            auth_token: Optional auth token for webhook.

        Returns:
            Dictionary with configuration information.

        Raises:
            A2AConnectionError: If connection to agent fails.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers=self._headers
            ) as httpx_client:
                if not self._agent_card:
                    msg = "Agent card not initialized"
                    raise A2AConfigurationError(msg)

                if not self._transport_protocol:
                    msg = "Transport protocol not configured"
                    raise A2AConfigurationError(msg)

                config = ClientConfig(
                    httpx_client=httpx_client,
                    supported_transports=[str(self._transport_protocol.value)],
                    streaming=self.enable_streaming,
                )

                factory = ClientFactory(config)
                client = factory.create(self._agent_card)

                push_config = PushNotificationConfig(
                    url=webhook_url,
                    token=auth_token,
                    authentication=(
                        PushNotificationAuthenticationInfo(
                            schemes=["bearer"],
                            credentials={"token": auth_token} if auth_token else {},
                        )
                        if auth_token
                        else None
                    ),
                )

                callback_config = TaskPushNotificationConfig(
                    task_id=task_id,
                    push_notification_config=push_config,
                )
                response = await client.set_task_callback(callback_config)

                result = {
                    "config_id": response.task_id,
                    "url": response.push_notification_config.url
                    if response.push_notification_config
                    else None,
                    "task_id": task_id,
                }

                self._logger.log(
                    "info", f"Webhook configured successfully for task {task_id}"
                )
                return result

        except Exception as e:
            self._logger.log(
                "error", f"Failed to configure webhook for task {task_id}: {e}"
            )
            raise A2AConnectionError(f"Failed to configure webhook: {e}") from e

    def get_task_callback(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve push notification webhook configuration for a task.

        Args:
            task_id: The ID of the task to retrieve webhook config for.

        Returns:
            Dictionary containing the webhook configuration, or None if not configured.
            Dictionary keys:
            - config_id: The configuration identifier
            - url: The webhook URL
            - task_id: The associated task ID

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2AConfigurationError: If agent card is not initialized.

        Example:
            ```python
            config = adapter.get_task_callback("task-123")
            if config:
                print(f"Webhook URL: {config['url']}")
            else:
                print("No webhook configured")
            ```
        """
        if not self._agent_card:
            msg = "Agent card not initialized. Call create_agent_executor() first."
            raise A2AConfigurationError(msg)

        self._logger.log("info", f"Retrieving webhook config for task: {task_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_task_callback_async(task_id))
        finally:
            loop.close()

    async def _get_task_callback_async(self, task_id: str) -> dict[str, Any] | None:
        """Async implementation of webhook configuration retrieval.

        Args:
            task_id: The ID of the task.

        Returns:
            Dictionary with configuration or None.

        Raises:
            A2AConnectionError: If connection to agent fails.
        """
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout, headers=self._headers
            ) as httpx_client:
                if not self._agent_card:
                    msg = "Agent card not initialized"
                    raise A2AConfigurationError(msg)

                if not self._transport_protocol:
                    msg = "Transport protocol not configured"
                    raise A2AConfigurationError(msg)

                config = ClientConfig(
                    httpx_client=httpx_client,
                    supported_transports=[str(self._transport_protocol.value)],
                    streaming=self.enable_streaming,
                )

                factory = ClientFactory(config)
                client = factory.create(self._agent_card)

                params = GetTaskPushNotificationConfigParams(id=task_id)
                response = await client.get_task_callback(params)

                if not response.push_notification_config:
                    self._logger.log(
                        "info", f"No webhook configured for task {task_id}"
                    )
                    return None

                result = {
                    "config_id": response.task_id,
                    "url": response.push_notification_config.url,
                    "task_id": task_id,
                }

                self._logger.log("info", f"Retrieved webhook config for task {task_id}")
                return result

        except Exception as e:
            self._logger.log(
                "error", f"Failed to retrieve webhook config for task {task_id}: {e}"
            )
            raise A2AConnectionError(f"Failed to retrieve webhook config: {e}") from e

    def delete_task_callback(self, task_id: str, config_id: str | None = None) -> bool:
        """Delete push notification webhook configuration for a task.

        Args:
            task_id: The ID of the task to delete webhook config for.
            config_id: Optional configuration ID. If not provided, deletes all
                webhook configurations for the task.

        Returns:
            True if deletion was successful, False otherwise.

        Raises:
            A2AConnectionError: If connection to agent fails.
            A2AConfigurationError: If agent card is not initialized.

        Example:
            ```python
            success = adapter.delete_task_callback("task-123")
            if success:
                print("Webhook configuration deleted")
            ```
        """
        if not self._agent_card:
            msg = "Agent card not initialized. Call create_agent_executor() first."
            raise A2AConfigurationError(msg)

        self._logger.log("info", f"Deleting webhook config for task: {task_id}")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._delete_task_callback_async(task_id, config_id)
            )
        finally:
            loop.close()

    async def _delete_task_callback_async(
        self,
        task_id: str,
        config_id: str | None,
    ) -> bool:
        """Async implementation of webhook configuration deletion.

        Args:
            task_id: The ID of the task.
            config_id: Optional configuration ID.

        Returns:
            True if successful, False otherwise.

        Raises:
            A2AConnectionError: If connection to agent fails.
        """
        # Note: delete_task_callback is not yet available in current a2a-sdk versions
        # This method is provided for future compatibility
        msg = "delete_task_callback is not yet supported in current a2a-sdk version"
        self._logger.log("warning", msg)
        raise NotImplementedError(msg)

    def _check_io_mode_compatibility(self) -> None:
        """Check input/output mode compatibility and log warnings.

        Validates that the A2A agent supports text-based input/output,
        which is the only mode currently supported by this adapter.
        """
        if not self._agent_card:
            return

        if "text" not in self._agent_card.default_input_modes:
            self._logger.log(
                "warning",
                f"Agent prefers {self._agent_card.default_input_modes} input modes, "
                "but CrewAI only supports 'text'. Communication may be limited or fail.",
            )

        if "text" not in self._agent_card.default_output_modes:
            self._logger.log(
                "warning",
                f"Agent prefers {self._agent_card.default_output_modes} output modes, "
                "but CrewAI only supports 'text'. Response parsing may be limited.",
            )

    def _check_state_transition_history(self) -> None:
        """Check if agent supports state transition history.

        Logs whether the agent tracks full history of task state transitions,
        which can be useful for debugging, monitoring, and auditing.
        """
        if not self._agent_card:
            return

        if self._agent_card.capabilities.state_transition_history:
            self._logger.log(
                "debug", "Agent supports state transition history tracking"
            )

    def get_delegation_tools(self, agents: list[BaseAgent]) -> list[BaseTool]:
        """Get delegation tools for A2A agents.

        A2A agents don't support CrewAI-style delegation as they manage
        their own agent interactions via the A2A protocol.

        Args:
            agents: List of agents in the crew.

        Returns:
            Empty list (A2A agents handle their own delegation).
        """
        return []

    def get_platform_tools(self) -> list[BaseTool]:
        """Get platform-specific tools for A2A agents.

        Currently, no platform-specific tools are provided for A2A agents.

        Returns:
            Empty list (no platform-specific tools for A2A agents).
        """
        return []
