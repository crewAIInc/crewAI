"""Utility functions for crew operations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from opentelemetry import baggage

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.crews.crew_output import CrewOutput
from crewai.rag.embeddings.types import EmbedderConfig
from crewai.types.streaming import CrewStreamingOutput, FlowStreamingOutput
from crewai.utilities.file_store import store_files
from crewai.utilities.streaming import (
    StreamingState,
    TaskInfo,
    create_streaming_state,
)


try:
    from crewai_files import (
        AudioFile,
        ImageFile,
        PDFFile,
        TextFile,
        VideoFile,
    )

    _FILE_TYPES: tuple[type, ...] = (AudioFile, ImageFile, PDFFile, TextFile, VideoFile)
except ImportError:
    _FILE_TYPES = ()


if TYPE_CHECKING:
    from crewai_files import FileInput

    from crewai.crew import Crew


def enable_agent_streaming(agents: Iterable[BaseAgent]) -> None:
    """Enable streaming on all agents that have an LLM configured.

    Args:
        agents: Iterable of agents to enable streaming on.
    """
    for agent in agents:
        if agent.llm is not None:
            agent.llm.stream = True


def setup_agents(
    crew: Crew,
    agents: Iterable[BaseAgent],
    embedder: EmbedderConfig | None,
    function_calling_llm: Any,
    step_callback: Callable[..., Any] | None,
) -> None:
    """Set up agents for crew execution.

    Args:
        crew: The crew instance agents belong to.
        agents: Iterable of agents to set up.
        embedder: Embedder configuration for knowledge.
        function_calling_llm: Default function calling LLM for agents.
        step_callback: Default step callback for agents.
    """
    for agent in agents:
        agent.crew = crew
        agent.set_knowledge(crew_embedder=embedder)
        if not agent.function_calling_llm:  # type: ignore[attr-defined]
            agent.function_calling_llm = function_calling_llm  # type: ignore[attr-defined]
        if not agent.step_callback:  # type: ignore[attr-defined]
            agent.step_callback = step_callback  # type: ignore[attr-defined]
        agent.create_agent_executor()


class TaskExecutionData:
    """Data container for prepared task execution information."""

    def __init__(
        self,
        agent: BaseAgent | None,
        tools: list[Any],
        should_skip: bool = False,
    ) -> None:
        """Initialize task execution data.

        Args:
            agent: The agent to use for task execution (None if skipped).
            tools: Prepared tools for the task.
            should_skip: Whether the task should be skipped (replay).
        """
        self.agent = agent
        self.tools = tools
        self.should_skip = should_skip


def prepare_task_execution(
    crew: Crew,
    task: Any,
    task_index: int,
    start_index: int | None,
    task_outputs: list[Any],
    last_sync_output: Any | None,
) -> tuple[TaskExecutionData, list[Any], Any | None]:
    """Prepare a task for execution, handling replay skip logic and agent/tool setup.

    Args:
        crew: The crew instance.
        task: The task to prepare.
        task_index: Index of the current task.
        start_index: Index to start execution from (for replay).
        task_outputs: Current list of task outputs.
        last_sync_output: Last synchronous task output.

    Returns:
        A tuple of (TaskExecutionData or None if skipped, updated task_outputs, updated last_sync_output).
        If the task should be skipped, TaskExecutionData will have should_skip=True.

    Raises:
        ValueError: If no agent is available for the task.
    """
    # Handle replay skip
    if start_index is not None and task_index < start_index:
        if task.output:
            if task.async_execution:
                task_outputs.append(task.output)
            else:
                task_outputs = [task.output]
                last_sync_output = task.output
        return (
            TaskExecutionData(agent=None, tools=[], should_skip=True),
            task_outputs,
            last_sync_output,
        )

    agent_to_use = crew._get_agent_to_use(task)
    if agent_to_use is None:
        raise ValueError(
            f"No agent available for task: {task.description}. "
            f"Ensure that either the task has an assigned agent "
            f"or a manager agent is provided."
        )

    tools_for_task = task.tools or agent_to_use.tools or []
    tools_for_task = crew._prepare_tools(
        agent_to_use,
        task,
        tools_for_task,
    )

    crew._log_task_start(task, agent_to_use.role)

    return (
        TaskExecutionData(agent=agent_to_use, tools=tools_for_task),
        task_outputs,
        last_sync_output,
    )


def check_conditional_skip(
    crew: Crew,
    task: Any,
    task_outputs: list[Any],
    task_index: int,
    was_replayed: bool,
) -> Any | None:
    """Check if a conditional task should be skipped.

    Args:
        crew: The crew instance.
        task: The conditional task to check.
        task_outputs: List of previous task outputs.
        task_index: Index of the current task.
        was_replayed: Whether this is a replayed execution.

    Returns:
        The skipped task output if the task should be skipped, None otherwise.
    """
    previous_output = task_outputs[-1] if task_outputs else None
    if previous_output is not None and not task.should_execute(previous_output):
        crew._logger.log(
            "debug",
            f"Skipping conditional task: {task.description}",
            color="yellow",
        )
        skipped_task_output = task.get_skipped_task_output()

        if not was_replayed:
            crew._store_execution_log(task, skipped_task_output, task_index)
        return skipped_task_output
    return None


def _extract_files_from_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Extract file objects from inputs dict.

    Scans inputs for FileInput objects (ImageFile, TextFile, etc.) and
    extracts them into a separate dict.

    Args:
        inputs: The inputs dictionary to scan.

    Returns:
        Dictionary of extracted file objects.
    """
    if not _FILE_TYPES:
        return {}

    files: dict[str, Any] = {}
    keys_to_remove: list[str] = []

    for key, value in inputs.items():
        if isinstance(value, _FILE_TYPES):
            files[key] = value
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del inputs[key]

    return files


def prepare_kickoff(
    crew: Crew,
    inputs: dict[str, Any] | None,
    input_files: dict[str, FileInput] | None = None,
) -> dict[str, Any] | None:
    """Prepare crew for kickoff execution.

    Handles before callbacks, event emission, task handler reset, input
    interpolation, task callbacks, agent setup, and planning.

    Args:
        crew: The crew instance to prepare.
        inputs: Optional input dictionary to pass to the crew.
        input_files: Optional dict of named file inputs for the crew.

    Returns:
        The potentially modified inputs dictionary after before callbacks.
    """
    from crewai.events.base_events import reset_emission_counter
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.event_context import get_current_parent_id, reset_last_event_id
    from crewai.events.types.crew_events import CrewKickoffStartedEvent

    if get_current_parent_id() is None:
        reset_emission_counter()
        reset_last_event_id()

    # Normalize inputs to dict[str, Any] for internal processing
    normalized: dict[str, Any] | None = None
    if inputs is not None:
        if not isinstance(inputs, Mapping):
            raise TypeError(
                f"inputs must be a dict or Mapping, got {type(inputs).__name__}"
            )
        normalized = dict(inputs)

    for before_callback in crew.before_kickoff_callbacks:
        if normalized is None:
            normalized = {}
        normalized = before_callback(normalized)

    future = crewai_event_bus.emit(
        crew,
        CrewKickoffStartedEvent(crew_name=crew.name, inputs=normalized),
    )
    if future is not None:
        try:
            future.result()
        except Exception:  # noqa: S110
            pass

    crew._task_output_handler.reset()
    crew._logging_color = "bold_purple"

    # Check for flow input files in baggage context (inherited from parent Flow)
    _flow_files = baggage.get_baggage("flow_input_files")
    flow_files: dict[str, Any] = _flow_files if isinstance(_flow_files, dict) else {}

    if normalized is not None:
        # Extract file objects unpacked directly into inputs
        unpacked_files = _extract_files_from_inputs(normalized)

        # Merge files: flow_files < input_files < unpacked_files (later takes precedence)
        all_files = {**flow_files, **(input_files or {}), **unpacked_files}
        if all_files:
            store_files(crew.id, all_files)

        crew._inputs = normalized
        crew._interpolate_inputs(normalized)
    else:
        # No inputs dict provided
        all_files = {**flow_files, **(input_files or {})}
        if all_files:
            store_files(crew.id, all_files)
    crew._set_tasks_callbacks()
    crew._set_allow_crewai_trigger_context_for_first_task()

    setup_agents(
        crew,
        crew.agents,
        crew.embedder,
        crew.function_calling_llm,
        crew.step_callback,
    )

    if crew.planning:
        crew._handle_crew_planning()

    return normalized


class StreamingContext:
    """Container for streaming state and holders used during crew execution."""

    def __init__(self, use_async: bool = False) -> None:
        """Initialize streaming context.

        Args:
            use_async: Whether to use async streaming mode.
        """
        self.result_holder: list[CrewOutput] = []
        self.current_task_info: TaskInfo = {
            "index": 0,
            "name": "",
            "id": "",
            "agent_role": "",
            "agent_id": "",
        }
        self.state: StreamingState = create_streaming_state(
            self.current_task_info, self.result_holder, use_async=use_async
        )
        self.output_holder: list[CrewStreamingOutput | FlowStreamingOutput] = []


class ForEachStreamingContext:
    """Container for streaming state used in for_each crew execution methods."""

    def __init__(self) -> None:
        """Initialize for_each streaming context."""
        self.result_holder: list[list[CrewOutput]] = [[]]
        self.current_task_info: TaskInfo = {
            "index": 0,
            "name": "",
            "id": "",
            "agent_role": "",
            "agent_id": "",
        }
        self.state: StreamingState = create_streaming_state(
            self.current_task_info, self.result_holder, use_async=True
        )
        self.output_holder: list[CrewStreamingOutput | FlowStreamingOutput] = []


async def run_for_each_async(
    crew: Crew,
    inputs: list[dict[str, Any]],
    kickoff_fn: Callable[
        [Crew, dict[str, Any]], Coroutine[Any, Any, CrewOutput | CrewStreamingOutput]
    ],
) -> list[CrewOutput | CrewStreamingOutput] | CrewStreamingOutput:
    """Execute crew workflow for each input asynchronously.

    Args:
        crew: The crew instance to execute.
        inputs: List of input dictionaries for each execution.
        kickoff_fn: Async function to call for each crew copy (kickoff_async or akickoff).

    Returns:
        If streaming, a single CrewStreamingOutput that yields chunks from all crews.
        Otherwise, a list of CrewOutput results.
    """
    from crewai.types.usage_metrics import UsageMetrics
    from crewai.utilities.streaming import (
        create_async_chunk_generator,
        signal_end,
        signal_error,
    )

    crew_copies = [crew.copy() for _ in inputs]

    if crew.stream:
        ctx = ForEachStreamingContext()

        async def run_all_crews() -> None:
            try:
                streaming_outputs: list[CrewStreamingOutput] = []
                for i, crew_copy in enumerate(crew_copies):
                    streaming = await kickoff_fn(crew_copy, inputs[i])
                    if isinstance(streaming, CrewStreamingOutput):
                        streaming_outputs.append(streaming)

                async def consume_stream(
                    stream_output: CrewStreamingOutput,
                ) -> CrewOutput:
                    async for chunk in stream_output:
                        if (
                            ctx.state.async_queue is not None
                            and ctx.state.loop is not None
                        ):
                            ctx.state.loop.call_soon_threadsafe(
                                ctx.state.async_queue.put_nowait, chunk
                            )
                    return stream_output.result

                crew_results = await asyncio.gather(
                    *[consume_stream(s) for s in streaming_outputs]
                )
                ctx.result_holder[0] = list(crew_results)
            except Exception as e:
                signal_error(ctx.state, e, is_async=True)
            finally:
                signal_end(ctx.state, is_async=True)

        streaming_output = CrewStreamingOutput(
            async_iterator=create_async_chunk_generator(
                ctx.state, run_all_crews, ctx.output_holder
            )
        )

        def set_results_wrapper(result: Any) -> None:
            streaming_output._set_results(result)

        streaming_output._set_result = set_results_wrapper  # type: ignore[method-assign]
        ctx.output_holder.append(streaming_output)

        return streaming_output

    async_tasks: list[asyncio.Task[CrewOutput | CrewStreamingOutput]] = [
        asyncio.create_task(kickoff_fn(crew_copy, input_data))
        for crew_copy, input_data in zip(crew_copies, inputs, strict=True)
    ]

    results = await asyncio.gather(*async_tasks)

    total_usage_metrics = UsageMetrics()
    for crew_copy in crew_copies:
        if crew_copy.usage_metrics:
            total_usage_metrics.add_usage_metrics(crew_copy.usage_metrics)
    crew.usage_metrics = total_usage_metrics

    crew._task_output_handler.reset()
    return list(results)
