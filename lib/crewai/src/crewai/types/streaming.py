"""Streaming output types for crew and flow execution."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from crewai.crews.crew_output import CrewOutput


T = TypeVar("T")


class StreamChunkType(Enum):
    """Type of streaming chunk."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    ERROR = "error"
    HEARTBEAT = "heartbeat"  # For keeping UI alive during long LLM waits


class ToolCallChunk(BaseModel):
    """Tool call information in a streaming chunk.

    Attributes:
        tool_id: Unique identifier for the tool call
        tool_name: Name of the tool being called
        arguments: JSON string of tool arguments
        index: Index of the tool call in the response
    """

    tool_id: str | None = None
    tool_name: str | None = None
    arguments: str = ""
    index: int = 0


class TaskInfoChunk(BaseModel):
    """Task information for lifecycle chunks.

    Attributes:
        task_index: Index of the task (0-based)
        task_name: Name or description of the task
        task_id: Unique identifier of the task
        expected_output: Expected output description
        total_tasks: Total number of tasks in the crew
        output: Task output (for TASK_COMPLETED chunks)
    """

    task_index: int = 0
    task_name: str = ""
    task_id: str = ""
    expected_output: str = ""
    total_tasks: int = 0
    output: str | None = None


class AgentInfoChunk(BaseModel):
    """Agent information for lifecycle chunks.

    Attributes:
        agent_role: Role of the agent
        agent_id: Unique identifier of the agent
        agent_goal: Goal of the agent
    """

    agent_role: str = ""
    agent_id: str = ""
    agent_goal: str = ""


class StreamChunk(BaseModel):
    """Base streaming chunk with full context.

    Attributes:
        content: The streaming content (text or partial content)
        chunk_type: Type of the chunk (text, tool_call, etc.)
        task_index: Index of the current task (0-based)
        task_name: Name or description of the current task
        task_id: Unique identifier of the task
        agent_role: Role of the agent executing the task
        agent_id: Unique identifier of the agent
        tool_call: Tool call information if chunk_type is TOOL_CALL
        task_info: Detailed task info for lifecycle chunks
        agent_info: Detailed agent info for lifecycle chunks
        timestamp: When this chunk was created
    """

    content: str = Field(description="The streaming content")
    chunk_type: StreamChunkType = Field(
        default=StreamChunkType.TEXT, description="Type of the chunk"
    )
    task_index: int = Field(default=0, description="Index of the current task")
    task_name: str = Field(default="", description="Name of the current task")
    task_id: str = Field(default="", description="Unique identifier of the task")
    agent_role: str = Field(default="", description="Role of the agent")
    agent_id: str = Field(default="", description="Unique identifier of the agent")
    tool_call: ToolCallChunk | None = Field(
        default=None, description="Tool call information"
    )
    task_info: TaskInfoChunk | None = Field(
        default=None, description="Detailed task info for lifecycle chunks"
    )
    agent_info: AgentInfoChunk | None = Field(
        default=None, description="Detailed agent info for lifecycle chunks"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, description="When this chunk was created"
    )

    def __str__(self) -> str:
        """Return the chunk content as a string."""
        return self.content


class UIOutputBuilder:
    """Builds UIOutput incrementally from stream chunks.

    This class processes StreamChunk objects and maintains the current
    state of the crew execution for real-time UI updates.

    Example:
        ```python
        # During streaming
        builder = UIOutputBuilder(crew)
        for chunk in streaming:
            builder.process_chunk(chunk)
            ui_state = builder.get_current_state()
            render_dashboard(ui_state)

        # After streaming
        ui_output = builder.finalize(crew_output)
        ```
    """

    def __init__(self, crew: Any) -> None:
        """Initialize the builder with crew information.

        Args:
            crew: The Crew instance being executed.
        """
        from crewai.types.ui_output import (
            AgentUIInfo,
            CrewUIInfo,
            TaskUIInfo,
            UIOutput,
        )

        self._crew = crew
        self._start_time = datetime.now()
        self._chunks: list[StreamChunk] = []
        self._current_task_index: int = 0
        self._task_start_times: dict[int, datetime] = {}
        self._task_end_times: dict[int, datetime] = {}
        self._task_outputs: dict[int, str] = {}

        # Build initial agent info
        self._agents: list[AgentUIInfo] = []
        for agent in crew.agents:
            self._agents.append(
                AgentUIInfo(
                    id=str(agent.id),
                    role=agent.role,
                    goal=agent.goal or "",
                    backstory=agent.backstory or "",
                )
            )

        # Build initial task info
        self._tasks: list[TaskUIInfo] = []
        for i, task in enumerate(crew.tasks):
            agent_info = None
            if task.agent:
                agent_info = AgentUIInfo(
                    id=str(task.agent.id),
                    role=task.agent.role,
                    goal=task.agent.goal or "",
                    backstory=task.agent.backstory or "",
                )
            self._tasks.append(
                TaskUIInfo(
                    id=str(task.id),
                    index=i,
                    name=task.name or task.description[:50] if task.description else "",
                    description=task.description or "",
                    expected_output=task.expected_output or "",
                    agent=agent_info,
                    status="pending",
                )
            )

        # Build crew info
        self._crew_info = CrewUIInfo(
            id=str(crew.id) if hasattr(crew, "id") else "",
            name=crew.name or "",
            process=str(crew.process.value) if crew.process else "",
            total_tasks=len(crew.tasks),
            completed_tasks=0,
            current_task_index=0,
            status="running",
        )

    def process_chunk(self, chunk: StreamChunk) -> None:
        """Process a stream chunk and update state.

        Args:
            chunk: The StreamChunk to process.
        """
        self._chunks.append(chunk)

        if chunk.chunk_type == StreamChunkType.TASK_STARTED:
            self._handle_task_started(chunk)
        elif chunk.chunk_type == StreamChunkType.TASK_COMPLETED:
            self._handle_task_completed(chunk)

    def _handle_task_started(self, chunk: StreamChunk) -> None:
        """Handle a TASK_STARTED chunk."""
        task_index = chunk.task_index
        self._current_task_index = task_index
        self._task_start_times[task_index] = chunk.timestamp
        self._crew_info.current_task_index = task_index

        if task_index < len(self._tasks):
            self._tasks[task_index].status = "in_progress"
            self._tasks[task_index].start_time = chunk.timestamp

    def _handle_task_completed(self, chunk: StreamChunk) -> None:
        """Handle a TASK_COMPLETED chunk."""
        task_index = chunk.task_index
        self._task_end_times[task_index] = chunk.timestamp
        self._crew_info.completed_tasks += 1

        if task_index < len(self._tasks):
            self._tasks[task_index].status = "completed"
            self._tasks[task_index].end_time = chunk.timestamp

            # Calculate duration
            if task_index in self._task_start_times:
                start = self._task_start_times[task_index]
                self._tasks[task_index].duration_seconds = (
                    chunk.timestamp - start
                ).total_seconds()

            # Store output
            if chunk.task_info and chunk.task_info.output:
                output = chunk.task_info.output
                self._tasks[task_index].output = output
                self._tasks[task_index].output_summary = (
                    output[:200] + "..." if len(output) > 200 else output
                )
                self._task_outputs[task_index] = output

    def get_current_state(self) -> Any:
        """Get the current UI state.

        Returns:
            UIOutput with the current execution state.
        """
        from crewai.types.ui_output import UIOutput

        # Determine current agent/task
        current_task = None
        current_agent = None
        if self._current_task_index < len(self._tasks):
            task = self._tasks[self._current_task_index]
            if task.status == "in_progress":
                current_task = task
                current_agent = task.agent

        execution_time = (datetime.now() - self._start_time).total_seconds()

        return UIOutput(
            crew=self._crew_info,
            agents=self._agents,
            tasks=self._tasks,
            current_agent=current_agent,
            current_task=current_task,
            last_updated=datetime.now(),
            raw_output=self.get_full_text(),
            execution_time_seconds=execution_time,
        )

    def get_full_text(self) -> str:
        """Get all streamed text content concatenated.

        Returns:
            All text chunks concatenated together.
        """
        return "".join(
            chunk.content
            for chunk in self._chunks
            if chunk.chunk_type == StreamChunkType.TEXT
        )

    def finalize(self, crew_output: Any) -> Any:
        """Finalize the UIOutput with the crew output.

        Args:
            crew_output: The final CrewOutput.

        Returns:
            Final UIOutput with complete data.
        """
        from crewai.types.ui_output import UIOutput

        self._crew_info.status = "completed"
        execution_time = (datetime.now() - self._start_time).total_seconds()

        # Update any remaining task info from crew_output
        if hasattr(crew_output, "tasks_output") and crew_output.tasks_output:
            for i, task_output in enumerate(crew_output.tasks_output):
                if i < len(self._tasks):
                    output_str = str(task_output.raw) if hasattr(task_output, "raw") else str(task_output)
                    self._tasks[i].output = output_str
                    self._tasks[i].output_summary = (
                        output_str[:200] + "..." if len(output_str) > 200 else output_str
                    )
                    self._tasks[i].status = "completed"

        return UIOutput(
            crew=self._crew_info,
            agents=self._agents,
            tasks=self._tasks,
            current_agent=None,
            current_task=None,
            last_updated=datetime.now(),
            raw_output=str(crew_output.raw) if hasattr(crew_output, "raw") else str(crew_output),
            execution_time_seconds=execution_time,
        )


class StreamingOutputBase(Generic[T]):
    """Base class for streaming output with result access.

    Provides iteration over stream chunks and access to final result
    via the .result property after streaming completes.
    """

    def __init__(self) -> None:
        """Initialize streaming output base."""
        self._result: T | None = None
        self._completed: bool = False
        self._chunks: list[StreamChunk] = []
        self._error: Exception | None = None

    @property
    def result(self) -> T:
        """Get the final result after streaming completes.

        Returns:
            The final output (CrewOutput for crews, Any for flows).

        Raises:
            RuntimeError: If streaming has not completed yet.
            Exception: If streaming failed with an error.
        """
        if not self._completed:
            raise RuntimeError(
                "Streaming has not completed yet. "
                "Iterate over all chunks before accessing result."
            )
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("No result available")
        return self._result

    @property
    def is_completed(self) -> bool:
        """Check if streaming has completed."""
        return self._completed

    @property
    def chunks(self) -> list[StreamChunk]:
        """Get all collected chunks so far."""
        return self._chunks.copy()

    def get_full_text(self) -> str:
        """Get all streamed text content concatenated.

        Returns:
            All text chunks concatenated together.
        """
        return "".join(
            chunk.content
            for chunk in self._chunks
            if chunk.chunk_type == StreamChunkType.TEXT
        )


class CrewStreamingOutput(StreamingOutputBase["CrewOutput"]):
    """Streaming output wrapper for crew execution.

    Provides both sync and async iteration over stream chunks,
    with access to the final CrewOutput via the .result property.

    For kickoff_for_each_async with streaming, use .results to get list of outputs.

    Example:
        ```python
        # Single crew
        streaming = crew.kickoff(inputs={"topic": "AI"})
        for chunk in streaming:
            print(chunk.content, end="", flush=True)
        result = streaming.result

        # With UI state tracking
        for chunk in streaming:
            print(chunk.content, end="", flush=True)
            # Get current UI state (during streaming)
            ui_state = streaming.get_ui_state()
            render_progress(ui_state.crew.completed_tasks, ui_state.crew.total_tasks)
        # Get final UI output
        ui_output = streaming.get_ui_output()

        # Multiple crews (kickoff_for_each_async)
        streaming = await crew.kickoff_for_each_async(
            [{"topic": "AI"}, {"topic": "ML"}]
        )
        async for chunk in streaming:
            print(chunk.content, end="", flush=True)
        results = streaming.results  # List of CrewOutput
        ```
    """

    def __init__(
        self,
        sync_iterator: Iterator[StreamChunk] | None = None,
        async_iterator: AsyncIterator[StreamChunk] | None = None,
        crew: Any = None,
    ) -> None:
        """Initialize crew streaming output.

        Args:
            sync_iterator: Synchronous iterator for chunks.
            async_iterator: Asynchronous iterator for chunks.
            crew: Optional crew instance for UI state tracking.
        """
        super().__init__()
        self._sync_iterator = sync_iterator
        self._async_iterator = async_iterator
        self._results: list[CrewOutput] | None = None
        self._crew = crew
        self._ui_builder: UIOutputBuilder | None = None
        if crew is not None:
            self._ui_builder = UIOutputBuilder(crew)

    @property
    def results(self) -> list[CrewOutput]:
        """Get all results for kickoff_for_each_async.

        Returns:
            List of CrewOutput from all crews.

        Raises:
            RuntimeError: If streaming has not completed or results not available.
        """
        if not self._completed:
            raise RuntimeError(
                "Streaming has not completed yet. "
                "Iterate over all chunks before accessing results."
            )
        if self._error is not None:
            raise self._error
        if self._results is not None:
            return self._results
        if self._result is not None:
            return [self._result]
        raise RuntimeError("No results available")

    def _set_results(self, results: list[CrewOutput]) -> None:
        """Set multiple results for kickoff_for_each_async.

        Args:
            results: List of CrewOutput from all crews.
        """
        self._results = results
        self._completed = True

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over stream chunks synchronously.

        Yields:
            StreamChunk objects as they arrive.

        Raises:
            RuntimeError: If sync iterator not available.
        """
        if self._sync_iterator is None:
            raise RuntimeError("Sync iterator not available")
        try:
            for chunk in self._sync_iterator:
                self._chunks.append(chunk)
                if self._ui_builder is not None:
                    self._ui_builder.process_chunk(chunk)
                yield chunk
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Return async iterator for stream chunks.

        Returns:
            Async iterator for StreamChunk objects.
        """
        return self._async_iterate()

    async def _async_iterate(self) -> AsyncIterator[StreamChunk]:
        """Iterate over stream chunks asynchronously.

        Yields:
            StreamChunk objects as they arrive.

        Raises:
            RuntimeError: If async iterator not available.
        """
        if self._async_iterator is None:
            raise RuntimeError("Async iterator not available")
        try:
            async for chunk in self._async_iterator:
                self._chunks.append(chunk)
                if self._ui_builder is not None:
                    self._ui_builder.process_chunk(chunk)
                yield chunk
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def _set_result(self, result: CrewOutput) -> None:
        """Set the final result after streaming completes.

        Args:
            result: The final CrewOutput.
        """
        self._result = result
        self._completed = True

    def get_ui_state(self) -> Any:
        """Get the current UI state during streaming.

        Returns the current execution state as a UIOutput object,
        useful for real-time UI updates during streaming.

        Returns:
            UIOutput with the current execution state.

        Raises:
            RuntimeError: If crew was not provided for UI tracking.
        """
        if self._ui_builder is None:
            raise RuntimeError(
                "UI tracking not available. "
                "Crew instance must be provided during initialization."
            )
        return self._ui_builder.get_current_state()

    def get_ui_output(self) -> Any:
        """Get the final UI output after streaming completes.

        Returns the complete UIOutput with all task results and timing.

        Returns:
            UIOutput with complete execution data.

        Raises:
            RuntimeError: If streaming has not completed or crew not provided.
        """
        if self._ui_builder is None:
            raise RuntimeError(
                "UI tracking not available. "
                "Crew instance must be provided during initialization."
            )
        if not self._completed:
            raise RuntimeError(
                "Streaming has not completed yet. "
                "Iterate over all chunks before accessing UI output."
            )
        if self._result is not None:
            return self._ui_builder.finalize(self._result)
        return self._ui_builder.get_current_state()


class FlowStreamingOutput(StreamingOutputBase[Any]):
    """Streaming output wrapper for flow execution.

    Provides both sync and async iteration over stream chunks,
    with access to the final flow output via the .result property.

    Example:
        ```python
        # Sync usage
        streaming = flow.kickoff_streaming()
        for chunk in streaming:
            print(chunk.content, end="", flush=True)
        result = streaming.result

        # Async usage
        streaming = await flow.kickoff_streaming_async()
        async for chunk in streaming:
            print(chunk.content, end="", flush=True)
        result = streaming.result
        ```
    """

    def __init__(
        self,
        sync_iterator: Iterator[StreamChunk] | None = None,
        async_iterator: AsyncIterator[StreamChunk] | None = None,
    ) -> None:
        """Initialize flow streaming output.

        Args:
            sync_iterator: Synchronous iterator for chunks.
            async_iterator: Asynchronous iterator for chunks.
        """
        super().__init__()
        self._sync_iterator = sync_iterator
        self._async_iterator = async_iterator

    def __iter__(self) -> Iterator[StreamChunk]:
        """Iterate over stream chunks synchronously.

        Yields:
            StreamChunk objects as they arrive.

        Raises:
            RuntimeError: If sync iterator not available.
        """
        if self._sync_iterator is None:
            raise RuntimeError("Sync iterator not available")
        try:
            for chunk in self._sync_iterator:
                self._chunks.append(chunk)
                yield chunk
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Return async iterator for stream chunks.

        Returns:
            Async iterator for StreamChunk objects.
        """
        return self._async_iterate()

    async def _async_iterate(self) -> AsyncIterator[StreamChunk]:
        """Iterate over stream chunks asynchronously.

        Yields:
            StreamChunk objects as they arrive.

        Raises:
            RuntimeError: If async iterator not available.
        """
        if self._async_iterator is None:
            raise RuntimeError("Async iterator not available")
        try:
            async for chunk in self._async_iterator:
                self._chunks.append(chunk)
                yield chunk
        except Exception as e:
            self._error = e
            raise
        finally:
            self._completed = True

    def _set_result(self, result: Any) -> None:
        """Set the final result after streaming completes.

        Args:
            result: The final flow output.
        """
        self._result = result
        self._completed = True
