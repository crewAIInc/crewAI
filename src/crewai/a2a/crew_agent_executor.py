"""CrewAI Agent Executor for A2A Protocol Integration.

This module implements the A2A AgentExecutor interface to enable CrewAI crews
to participate in the Agent-to-Agent protocol for remote interoperability.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from crewai import Crew
from crewai.crew import CrewOutput

try:
    from a2a.server.agent_execution.agent_executor import AgentExecutor
    from a2a.server.agent_execution.context import RequestContext
    from a2a.server.events.event_queue import EventQueue
    from a2a.types import (
        InvalidParamsError,
        Part,
        Task,
        TextPart,
        UnsupportedOperationError,
    )
    from a2a.utils import completed_task, new_artifact
    from a2a.utils.errors import ServerError
except ImportError:
    raise ImportError(
        "A2A integration requires the 'a2a' extra dependency. "
        "Install with: pip install crewai[a2a]"
    )

logger = logging.getLogger(__name__)


class A2AServerError(Exception):
    """Base exception for A2A server errors."""
    pass


class TransportError(A2AServerError):
    """Error related to transport configuration."""
    pass


class ExecutionError(A2AServerError):
    """Error during crew execution."""
    pass


@dataclass
class TaskInfo:
    """Information about a running task."""
    task: asyncio.Task
    started_at: datetime
    status: str = "running"


class CrewAgentExecutor(AgentExecutor):
    """A2A Agent Executor that wraps CrewAI crews for remote interoperability.
    
    This class implements the A2A AgentExecutor interface to enable CrewAI crews
    to be exposed as remotely interoperable agents following the A2A protocol.
    
    Args:
        crew: The CrewAI crew to expose as an A2A agent
        supported_content_types: List of supported content types for input
        
    Example:
        from crewai import Agent, Crew, Task
        from crewai.a2a import CrewAgentExecutor
        
        agent = Agent(role="Assistant", goal="Help users", backstory="Helpful AI")
        task = Task(description="Help with {query}", agent=agent)
        crew = Crew(agents=[agent], tasks=[task])
        
        executor = CrewAgentExecutor(crew)
    """
    
    def __init__(
        self, 
        crew: Crew,
        supported_content_types: Optional[list[str]] = None
    ):
        """Initialize the CrewAgentExecutor.
        
        Args:
            crew: The CrewAI crew to wrap
            supported_content_types: List of supported content types
        """
        self.crew = crew
        self.supported_content_types = supported_content_types or [
            'text', 'text/plain'
        ]
        self._running_tasks: Dict[str, TaskInfo] = {}
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the crew with the given context and publish results to event queue.
        
        This method extracts the user input from the request context, executes
        the CrewAI crew, and publishes the results as A2A artifacts.
        
        Args:
            context: The A2A request context containing task details
            event_queue: Queue for publishing execution events and results
            
        Raises:
            ServerError: If validation fails or execution encounters an error
        """
        error = self._validate_request(context)
        if error:
            logger.error(f"Request validation failed: {error}")
            raise ServerError(error=InvalidParamsError())
        
        query = context.get_user_input()
        task_id = context.task_id
        context_id = context.context_id
        
        if not task_id or not context_id:
            raise ServerError(error=InvalidParamsError())
        
        logger.info(f"Executing crew for task {task_id} with query: {query}")
        
        try:
            inputs = {"query": query}
            
            execution_task = asyncio.create_task(
                self._execute_crew_async(inputs)
            )
            self._running_tasks[task_id] = TaskInfo(
                task=execution_task,
                started_at=datetime.now(),
                status="running"
            )
            
            result = await execution_task
            
            self._running_tasks.pop(task_id, None)
            
            logger.info(f"Crew execution completed for task {task_id}")
            
            parts = self._convert_output_to_parts(result)
            
            messages = [context.message] if context.message else []
            event_queue.enqueue_event(
                completed_task(
                    task_id,
                    context_id,
                    [new_artifact(parts, f"crew_output_{task_id}")],
                    messages,
                )
            )
            
        except asyncio.CancelledError:
            logger.info(f"Task {task_id} was cancelled")
            self._running_tasks.pop(task_id, None)
            raise
        except Exception as e:
            logger.error(f"Error executing crew for task {task_id}: {e}")
            self._running_tasks.pop(task_id, None)
            
            error_parts = [
                Part(root=TextPart(text=f"Error executing crew: {str(e)}"))
            ]
            
            messages = [context.message] if context.message else []
            event_queue.enqueue_event(
                completed_task(
                    task_id,
                    context_id,
                    [new_artifact(error_parts, f"error_{task_id}")],
                    messages,
                )
            )
            
            raise ServerError(
                error=InvalidParamsError()
            ) from e
    
    async def cancel(
        self, 
        request: RequestContext, 
        event_queue: EventQueue
    ) -> Task | None:
        """Cancel a running crew execution.
        
        Args:
            request: The A2A request context for the task to cancel
            event_queue: Event queue for publishing cancellation events
            
        Returns:
            None (cancellation is handled internally)
            
        Raises:
            ServerError: If the task cannot be cancelled
        """
        task_id = request.task_id
        
        if task_id in self._running_tasks:
            task_info = self._running_tasks[task_id]
            task_info.task.cancel()
            task_info.status = "cancelled"
            
            try:
                await task_info.task
            except asyncio.CancelledError:
                logger.info(f"Successfully cancelled task {task_id}")
                pass
            
            self._running_tasks.pop(task_id, None)
            return None
        else:
            logger.warning(f"Task {task_id} not found for cancellation")
            raise ServerError(error=UnsupportedOperationError())
    
    async def _execute_crew_async(self, inputs: Dict[str, Any]) -> CrewOutput:
        """Execute the crew asynchronously.
        
        Args:
            inputs: Input parameters for the crew
            
        Returns:
            The crew execution output
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.crew.kickoff, inputs)
    
    def _convert_output_to_parts(self, result: CrewOutput) -> list[Part]:
        """Convert CrewAI output to A2A Parts.
        
        Args:
            result: The crew execution result
            
        Returns:
            List of A2A Parts representing the output
        """
        parts = []
        
        if hasattr(result, 'raw') and result.raw:
            parts.append(Part(root=TextPart(text=str(result.raw))))
        elif result:
            parts.append(Part(root=TextPart(text=str(result))))
        
        if hasattr(result, 'json_dict') and result.json_dict:
            json_output = json.dumps(result.json_dict, indent=2)
            parts.append(Part(root=TextPart(text=json_output)))
        
        if not parts:
            parts.append(Part(root=TextPart(text="Crew execution completed successfully")))
        
        return parts
    
    def _validate_request(self, context: RequestContext) -> Optional[str]:
        """Validate the incoming request context.
        
        Args:
            context: The A2A request context to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        try:
            user_input = context.get_user_input()
            if not user_input or not user_input.strip():
                return "Empty or missing user input"
            
            return None
            
        except Exception as e:
            return f"Failed to extract user input: {e}"
