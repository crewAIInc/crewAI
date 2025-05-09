"""
A2A protocol agent integration for CrewAI.

This module implements the integration between CrewAI agents and the A2A protocol.
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from crewai.a2a.client import A2AClient
from crewai.a2a.task_manager import TaskManager
from crewai.types.a2a import (
    Artifact,
    DataPart,
    FilePart,
    Message,
    Part,
    Task as A2ATask,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatusUpdateEvent,
    TextPart,
)


class A2AAgentIntegration:
    """Integration between CrewAI agents and the A2A protocol."""

    def __init__(
        self,
        task_manager: Optional[TaskManager] = None,
        client: Optional[A2AClient] = None,
    ):
        """Initialize the A2A agent integration.

        Args:
            task_manager: The task manager to use for handling A2A tasks.
            client: The A2A client to use for sending tasks to other agents.
        """
        self.task_manager = task_manager
        self.client = client
        self.logger = logging.getLogger(__name__)

    async def execute_task_via_a2a(
        self,
        agent_url: str,
        task_description: str,
        context: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 300,
    ) -> str:
        """Execute a task via the A2A protocol.

        Args:
            agent_url: The URL of the agent to execute the task.
            task_description: The description of the task.
            context: Additional context for the task.
            api_key: The API key to use for authentication.
            timeout: The timeout for the task execution in seconds.

        Returns:
            The result of the task execution.

        Raises:
            TimeoutError: If the task execution times out.
            Exception: If there is an error executing the task.
        """
        if not self.client:
            self.client = A2AClient(base_url=agent_url, api_key=api_key)

        parts: List[Part] = [TextPart(text=task_description)]
        if context:
            parts.append(
                DataPart(
                    data={"context": context},
                    metadata={"type": "context"},
                )
            )

        message = Message(role="user", parts=parts)

        task_id = str(uuid.uuid4())

        try:
            queue = await self.client.send_task_streaming(
                task_id=task_id,
                message=message,
            )

            result = await self._wait_for_task_completion(queue, timeout)
            return result
        except Exception as e:
            self.logger.exception(f"Error executing task via A2A: {e}")
            raise

    async def _wait_for_task_completion(
        self, queue: asyncio.Queue, timeout: int
    ) -> str:
        """Wait for a task to complete.

        Args:
            queue: The queue to receive task updates from.
            timeout: The timeout for the task execution in seconds.

        Returns:
            The result of the task execution.

        Raises:
            TimeoutError: If the task execution times out.
            Exception: If there is an error executing the task.
        """
        result = ""
        try:
            async def _timeout():
                await asyncio.sleep(timeout)
                await queue.put(TimeoutError(f"Task execution timed out after {timeout} seconds"))

            timeout_task = asyncio.create_task(_timeout())

            while True:
                event = await queue.get()

                if isinstance(event, Exception):
                    raise event

                if isinstance(event, TaskStatusUpdateEvent):
                    if event.status.state == TaskState.COMPLETED:
                        if event.status.message:
                            for part in event.status.message.parts:
                                if isinstance(part, TextPart):
                                    result += part.text
                        break
                    elif event.status.state in [TaskState.FAILED, TaskState.CANCELED]:
                        error_message = "Task failed"
                        if event.status.message:
                            for part in event.status.message.parts:
                                if isinstance(part, TextPart):
                                    error_message = part.text
                        raise Exception(error_message)
                elif isinstance(event, TaskArtifactUpdateEvent):
                    for part in event.artifact.parts:
                        if isinstance(part, TextPart):
                            result += part.text
        finally:
            timeout_task.cancel()

        return result

    async def handle_a2a_task(
        self,
        task: A2ATask,
        agent_execute_func: Any,
        context: Optional[str] = None,
    ) -> None:
        """Handle an A2A task.

        Args:
            task: The A2A task to handle.
            agent_execute_func: The function to execute the task.
            context: Additional context for the task.

        Raises:
            Exception: If there is an error handling the task.
        """
        if not self.task_manager:
            raise ValueError("Task manager is required to handle A2A tasks")

        try:
            await self.task_manager.update_task_status(
                task_id=task.id,
                state=TaskState.WORKING,
            )

            task_description = ""
            task_context = context or ""

            if task.history and task.history[-1].role == "user":
                message = task.history[-1]
                for part in message.parts:
                    if isinstance(part, TextPart):
                        task_description += part.text
                    elif isinstance(part, DataPart) and part.data.get("context"):
                        task_context += part.data["context"]

            try:
                result = await agent_execute_func(task_description, task_context)

                response_message = Message(
                    role="agent",
                    parts=[TextPart(text=result)],
                )

                await self.task_manager.update_task_status(
                    task_id=task.id,
                    state=TaskState.COMPLETED,
                    message=response_message,
                )

                artifact = Artifact(
                    name="result",
                    parts=[TextPart(text=result)],
                )
                await self.task_manager.add_task_artifact(
                    task_id=task.id,
                    artifact=artifact,
                )
            except Exception as e:
                error_message = Message(
                    role="agent",
                    parts=[TextPart(text=str(e))],
                )
                await self.task_manager.update_task_status(
                    task_id=task.id,
                    state=TaskState.FAILED,
                    message=error_message,
                )
                raise
        except Exception as e:
            self.logger.exception(f"Error handling A2A task: {e}")
            raise
