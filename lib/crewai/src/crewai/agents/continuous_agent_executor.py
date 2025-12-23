"""Continuous agent executor for always-on operation mode.

This executor extends CrewAgentExecutor to support continuous operation
where agents run indefinitely until explicitly stopped.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
)
from crewai.continuous.state import ContinuousContext, ContinuousState
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.continuous_events import (
    ContinuousAgentActionEvent,
    ContinuousAgentObservationEvent,
    ContinuousErrorEvent,
    ContinuousIterationCompleteEvent,
)
from crewai.utilities.agent_utils import (
    enforce_rpm_limit,
    format_message_for_llm,
    get_llm_response,
    handle_context_length,
    handle_output_parser_exception,
    is_context_length_exceeded,
    process_llm_response,
)
from crewai.utilities.tool_utils import execute_tool_and_check_finality

if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.tools_handler import ToolsHandler
    from crewai.continuous.shutdown import ShutdownController
    from crewai.crew import Crew
    from crewai.llms.base_llm import BaseLLM
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool
    from crewai.types.continuous_streaming import ContinuousStreamingOutput
    from crewai.utilities.prompts import StandardPromptResult, SystemPromptResult


class AgentCheckpoint:
    """Represents a checkpoint in continuous execution.

    Unlike AgentFinish which signals completion, a checkpoint
    is a summary point that allows the agent to continue.
    """

    def __init__(
        self,
        thought: str,
        summary: str,
        text: str,
        continue_monitoring: bool = True,
    ) -> None:
        self.thought = thought
        self.summary = summary
        self.text = text
        self.continue_monitoring = continue_monitoring


class ContinuousAgentExecutor(CrewAgentExecutor):
    """Executor for continuous agent operation.

    Extends CrewAgentExecutor to support continuous operation mode where
    agents run indefinitely until explicitly stopped. Key differences:

    1. Never returns AgentFinish - converts to AgentCheckpoint
    2. Supports pause/resume functionality
    3. Emits continuous-specific events
    4. Manages memory to prevent unbounded growth
    5. Integrates with streaming output

    Example:
        ```python
        executor = ContinuousAgentExecutor(
            llm=llm,
            task=continuous_task,
            agent=agent,
            shutdown_controller=controller,
            state_manager=context,
            ...
        )

        # Initialize once
        executor.initialize(inputs)

        # Iterate continuously
        while not controller.should_stop:
            executor.continuous_iterate()
            time.sleep(1)
        ```
    """

    def __init__(
        self,
        llm: "BaseLLM",
        task: "Task",
        crew: "Crew",
        agent: "Agent",
        prompt: "SystemPromptResult | StandardPromptResult",
        max_iter: int,
        tools: list["CrewStructuredTool"],
        tools_names: str,
        stop_words: list[str],
        tools_description: str,
        tools_handler: "ToolsHandler",
        shutdown_controller: "ShutdownController",
        state_manager: ContinuousContext,
        streaming_output: "ContinuousStreamingOutput | None" = None,
        step_callback: Any = None,
        original_tools: list["BaseTool"] | None = None,
        function_calling_llm: "BaseLLM | Any | None" = None,
        respect_context_window: bool = True,
        request_within_rpm_limit: Callable[[], bool] | None = None,
        callbacks: list[Any] | None = None,
        max_messages: int = 100,
        iteration_delay: float = 0.0,
    ) -> None:
        """Initialize continuous executor.

        Args:
            llm: Language model instance.
            task: Continuous task to execute.
            crew: Crew instance.
            agent: Agent to execute.
            prompt: Prompt templates.
            max_iter: Maximum iterations per cycle (not total).
            tools: Available tools.
            tools_names: Tool names string.
            stop_words: Stop word list.
            tools_description: Tool descriptions.
            tools_handler: Tool handler instance.
            shutdown_controller: Controller for shutdown coordination.
            state_manager: Context for state management.
            streaming_output: Optional streaming output handler.
            step_callback: Optional step callback.
            original_tools: Original tool list.
            function_calling_llm: Optional function calling LLM.
            respect_context_window: Respect context limits.
            request_within_rpm_limit: RPM limit check function.
            callbacks: Optional callbacks list.
            max_messages: Maximum messages to keep in memory.
            iteration_delay: Delay between iterations in seconds.
        """
        super().__init__(
            llm=llm,
            task=task,
            crew=crew,
            agent=agent,
            prompt=prompt,
            max_iter=max_iter,
            tools=tools,
            tools_names=tools_names,
            stop_words=stop_words,
            tools_description=tools_description,
            tools_handler=tools_handler,
            step_callback=step_callback,
            original_tools=original_tools,
            function_calling_llm=function_calling_llm,
            respect_context_window=respect_context_window,
            request_within_rpm_limit=request_within_rpm_limit,
            callbacks=callbacks,
        )

        self.shutdown_controller = shutdown_controller
        self.state_manager = state_manager
        self.streaming_output = streaming_output
        self.max_messages = max_messages
        self.iteration_delay = iteration_delay
        self._initialized = False
        self._iteration_start_time: float = 0.0
        self._actions_this_iteration: int = 0

    def initialize(self, inputs: dict[str, Any] | None = None) -> None:
        """Initialize the executor with prompts.

        This should be called once before starting continuous iteration.

        Args:
            inputs: Optional input variables for prompt formatting.
        """
        if self._initialized:
            return

        inputs = inputs or {}

        if "system" in self.prompt:
            system_prompt = self._format_prompt(
                cast(str, self.prompt.get("system", "")), inputs
            )
            user_prompt = self._format_prompt(
                cast(str, self.prompt.get("user", "")), inputs
            )
            self.messages.append(format_message_for_llm(system_prompt, role="system"))
            self.messages.append(format_message_for_llm(user_prompt))
        else:
            user_prompt = self._format_prompt(self.prompt.get("prompt", ""), inputs)
            self.messages.append(format_message_for_llm(user_prompt))

        self._show_start_logs()
        self._initialized = True

    def continuous_iterate(self) -> AgentAction | AgentCheckpoint | None:
        """Execute a single continuous iteration.

        This method:
        1. Checks shutdown and pause state
        2. Gets LLM response
        3. Handles actions or checkpoints
        4. Manages memory
        5. Emits events

        Returns:
            The result of this iteration (AgentAction, AgentCheckpoint, or None)
        """
        # Check shutdown
        if self.shutdown_controller.should_stop:
            return None

        # Check pause state
        if self.state_manager.state == ContinuousState.PAUSED:
            time.sleep(0.1)  # Small delay while paused
            return None

        # Start iteration timing
        self._iteration_start_time = time.time()
        self._actions_this_iteration = 0

        try:
            # Enforce rate limits
            enforce_rpm_limit(self.request_within_rpm_limit)

            # Get LLM response
            answer = get_llm_response(
                llm=self.llm,
                messages=self.messages,
                callbacks=self.callbacks,
                printer=self._printer,
                from_task=self.task,
                from_agent=self.agent,
                response_model=None,  # No structured output in continuous mode
                executor_context=self,
            )
            formatted_answer = process_llm_response(answer, self.use_stop_words)

            result: AgentAction | AgentCheckpoint | None = None

            if isinstance(formatted_answer, AgentAction):
                # Handle tool use
                result = self._handle_continuous_action(formatted_answer)
                self._actions_this_iteration += 1

            elif isinstance(formatted_answer, AgentFinish):
                # Convert to checkpoint - don't actually finish
                result = self._handle_checkpoint(formatted_answer)

            # Update message history
            if formatted_answer:
                self._append_message(formatted_answer.text)

            # Manage memory to prevent unbounded growth
            self._manage_memory()

            # Update state
            self.state_manager.increment_iteration()
            self.iterations += 1

            # Emit iteration complete event
            self._emit_iteration_complete()

            # Apply iteration delay if configured
            if self.iteration_delay > 0:
                time.sleep(self.iteration_delay)

            return result

        except OutputParserError as e:
            handle_output_parser_exception(
                e=e,
                messages=self.messages,
                iterations=self.iterations,
                log_error_after=self.log_error_after,
                printer=self._printer,
            )
            self._emit_error(str(e), "OutputParserError", recoverable=True)
            return None

        except Exception as e:
            if is_context_length_exceeded(e):
                handle_context_length(
                    respect_context_window=self.respect_context_window,
                    printer=self._printer,
                    messages=self.messages,
                    llm=self.llm,
                    callbacks=self.callbacks,
                    i18n=self._i18n,
                )
                return None

            self._emit_error(str(e), type(e).__name__, recoverable=False)
            self.state_manager.record_error(str(e))
            raise

    def _handle_continuous_action(
        self, formatted_answer: AgentAction
    ) -> AgentAction:
        """Handle an agent action in continuous mode.

        Args:
            formatted_answer: The agent action to handle

        Returns:
            The processed agent action
        """
        # Extract fingerprint context if available
        fingerprint_context = {}
        if (
            self.agent
            and hasattr(self.agent, "security_config")
            and hasattr(self.agent.security_config, "fingerprint")
        ):
            fingerprint_context = {
                "agent_fingerprint": str(self.agent.security_config.fingerprint)
            }

        # Emit action event
        crewai_event_bus.emit(
            self.agent,
            ContinuousAgentActionEvent(
                agent_role=self.agent.role if self.agent else "",
                agent_id=str(self.agent.id) if self.agent else "",
                action_type="tool_use",
                tool_name=formatted_answer.tool,
                tool_input=self._parse_tool_input(formatted_answer.tool_input),
                thought=formatted_answer.thought,
                iteration=self.state_manager.iteration_count,
            ),
        )

        # Stream tool call if streaming is enabled
        if self.streaming_output:
            self.streaming_output.emit_tool_call(
                tool_name=formatted_answer.tool,
                tool_input=self._parse_tool_input(formatted_answer.tool_input),
                agent_role=self.agent.role if self.agent else "",
                agent_id=str(self.agent.id) if self.agent else "",
            )

        # Execute tool
        tool_result = execute_tool_and_check_finality(
            agent_action=formatted_answer,
            fingerprint_context=fingerprint_context,
            tools=self.tools,
            i18n=self._i18n,
            agent_key=self.agent.key if self.agent else None,
            agent_role=self.agent.role if self.agent else None,
            tools_handler=self.tools_handler,
            task=self.task,
            agent=self.agent,
            function_calling_llm=self.function_calling_llm,
            crew=self.crew,
        )

        # Emit observation event
        crewai_event_bus.emit(
            self.agent,
            ContinuousAgentObservationEvent(
                agent_role=self.agent.role if self.agent else "",
                agent_id=str(self.agent.id) if self.agent else "",
                observation=str(tool_result.result)[:500],  # Truncate for event
                triggered_by=formatted_answer.tool,
                iteration=self.state_manager.iteration_count,
            ),
        )

        # Stream observation if streaming is enabled
        if self.streaming_output:
            self.streaming_output.emit_observation(
                observation=str(tool_result.result),
                agent_role=self.agent.role if self.agent else "",
                agent_id=str(self.agent.id) if self.agent else "",
            )

        # Process action result
        formatted_answer.text += f"\nObservation: {tool_result.result}"
        formatted_answer.result = tool_result.result

        # In continuous mode, we don't treat result_as_answer as termination
        # Instead, we just continue monitoring

        self._invoke_step_callback(formatted_answer)
        return formatted_answer

    def _handle_checkpoint(self, formatted_answer: AgentFinish) -> AgentCheckpoint:
        """Convert AgentFinish to AgentCheckpoint for continuous mode.

        In continuous mode, we don't actually finish. Instead, we treat
        the "final answer" as a checkpoint/summary and continue monitoring.

        Args:
            formatted_answer: The agent finish to convert

        Returns:
            An AgentCheckpoint that allows continuing
        """
        checkpoint = AgentCheckpoint(
            thought=formatted_answer.thought,
            summary=formatted_answer.output,
            text=formatted_answer.text,
            continue_monitoring=True,
        )

        # Emit checkpoint event
        crewai_event_bus.emit(
            self.agent,
            ContinuousAgentActionEvent(
                agent_role=self.agent.role if self.agent else "",
                agent_id=str(self.agent.id) if self.agent else "",
                action_type="checkpoint",
                thought=checkpoint.thought,
                iteration=self.state_manager.iteration_count,
            ),
        )

        # Add checkpoint message to continue the conversation
        self._append_message(
            f"Checkpoint summary: {checkpoint.summary}\n\n"
            "Continue monitoring and take action as needed."
        )

        return checkpoint

    def _manage_memory(self) -> None:
        """Manage message memory to prevent unbounded growth.

        Keeps only the most recent messages up to max_messages limit.
        Preserves the system message if present.
        """
        if len(self.messages) <= self.max_messages:
            return

        # Always keep the first message (system prompt)
        system_message = None
        if self.messages and self.messages[0].get("role") == "system":
            system_message = self.messages[0]

        # Keep the most recent messages
        recent_messages = self.messages[-(self.max_messages - 1):]

        # Reconstruct messages list
        if system_message:
            self.messages = [system_message] + recent_messages
        else:
            self.messages = recent_messages[-self.max_messages:]

    def _emit_iteration_complete(self) -> None:
        """Emit iteration complete event."""
        duration = time.time() - self._iteration_start_time

        crewai_event_bus.emit(
            self.agent,
            ContinuousIterationCompleteEvent(
                iteration=self.state_manager.iteration_count,
                agents_active=[self.agent.role] if self.agent else [],
                actions_taken=self._actions_this_iteration,
                duration_seconds=duration,
            ),
        )

    def _emit_error(
        self,
        error: str,
        error_type: str,
        recoverable: bool = True
    ) -> None:
        """Emit error event.

        Args:
            error: Error message
            error_type: Type of error
            recoverable: Whether the error is recoverable
        """
        crewai_event_bus.emit(
            self.crew,
            ContinuousErrorEvent(
                crew_name=self.crew.name if self.crew else None,
                crew=self.crew,
                error=error,
                error_type=error_type,
                iteration=self.state_manager.iteration_count,
                agent_role=self.agent.role if self.agent else None,
                recoverable=recoverable,
            ),
        )

    def _parse_tool_input(self, tool_input: str | dict | None) -> dict[str, Any] | None:
        """Parse tool input to dictionary.

        Args:
            tool_input: Raw tool input (string or dict)

        Returns:
            Parsed dictionary or None
        """
        if tool_input is None:
            return None
        if isinstance(tool_input, dict):
            return tool_input
        try:
            import json
            return json.loads(tool_input)
        except (json.JSONDecodeError, TypeError):
            return {"raw": tool_input}

    def reset(self) -> None:
        """Reset the executor for a fresh start.

        Clears messages and iteration count but keeps configuration.
        """
        self.messages = []
        self.iterations = 0
        self._initialized = False
        self._actions_this_iteration = 0
