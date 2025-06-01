from typing import Any, Callable, Dict, List, Optional, Union, cast

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from crewai.agents.agent_state import AgentState
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from crewai.agents.tools_handler import ToolsHandler
from crewai.llm import BaseLLM
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_types import ToolResult
from crewai.utilities import I18N, Printer
from crewai.utilities.agent_utils import (
    enforce_rpm_limit,
    format_message_for_llm,
    get_llm_response,
    handle_agent_action_core,
    handle_context_length,
    handle_max_iterations_exceeded,
    handle_output_parser_exception,
    handle_unknown_error,
    has_reached_max_iterations,
    is_context_length_exceeded,
    process_llm_response,
    show_agent_logs,
)
from crewai.utilities.constants import MAX_LLM_RETRY, TRAINING_DATA_FILE
from crewai.utilities.logger import Logger
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from crewai.utilities.training_handler import CrewTrainingHandler


class CrewAgentExecutor(CrewAgentExecutorMixin):
    _logger: Logger = Logger()

    def __init__(
        self,
        llm: Any,
        task: Any,
        crew: Any,
        agent: BaseAgent,
        prompt: dict[str, str],
        max_iter: int,
        tools: List[CrewStructuredTool],
        tools_names: str,
        stop_words: List[str],
        tools_description: str,
        tools_handler: ToolsHandler,
        step_callback: Any = None,
        original_tools: List[Any] = [],
        function_calling_llm: Any = None,
        respect_context_window: bool = False,
        request_within_rpm_limit: Optional[Callable[[], bool]] = None,
        callbacks: List[Any] = [],
    ):
        self._i18n: I18N = I18N()
        self.llm: BaseLLM = llm
        self.task = task
        self.agent = agent
        self.crew = crew
        self.prompt = prompt
        self.tools = tools
        self.tools_names = tools_names
        self.stop = stop_words
        self.max_iter = max_iter
        self.callbacks = callbacks
        self._printer: Printer = Printer()
        self.tools_handler = tools_handler
        self.original_tools = original_tools
        self.step_callback = step_callback
        self.use_stop_words = self.llm.supports_stop_words()
        self.tools_description = tools_description
        self.function_calling_llm = function_calling_llm
        self.respect_context_window = respect_context_window
        self.request_within_rpm_limit = request_within_rpm_limit
        self.ask_for_human_input = False
        self.messages: List[Dict[str, str]] = []
        self.iterations = 0
        self.log_error_after = 3
        self.tool_name_to_tool_map: Dict[str, Union[CrewStructuredTool, BaseTool]] = {
            tool.name: tool for tool in self.tools
        }
        self.steps_since_reasoning = 0
        self.agent_state: AgentState = AgentState(task_id=str(task.id) if task else None)
        existing_stop = self.llm.stop or []
        self.llm.stop = list(
            set(
                existing_stop + self.stop
                if isinstance(existing_stop, list)
                else self.stop
            )
        )

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        # Reset agent state for new task execution
        self.agent_state.reset(task_id=str(self.task.id) if self.task else None)

        if "system" in self.prompt:
            system_prompt = self._format_prompt(self.prompt.get("system", ""), inputs)
            user_prompt = self._format_prompt(self.prompt.get("user", ""), inputs)
            self.messages.append(format_message_for_llm(system_prompt, role="system"))
            self.messages.append(format_message_for_llm(user_prompt))
        else:
            user_prompt = self._format_prompt(self.prompt.get("prompt", ""), inputs)
            self.messages.append(format_message_for_llm(user_prompt))

        self._show_start_logs()

        self.ask_for_human_input = bool(inputs.get("ask_for_human_input", False))

        try:
            # Populate agent state from reasoning output if available
            if hasattr(self.agent, "reasoning") and self.agent.reasoning:
                self._populate_state_from_reasoning()

            formatted_answer = self._invoke_loop()
        except AssertionError:
            self._printer.print(
                content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                color="red",
            )
            raise
        except Exception as e:
            handle_unknown_error(self._printer, e)
            if e.__class__.__module__.startswith("litellm"):
                # Do not retry on litellm errors
                raise e
            else:
                raise e

        if self.ask_for_human_input:
            formatted_answer = self._handle_human_feedback(formatted_answer)

        # Mark task as completed in agent state
        self.agent_state.mark_completed()

        self._create_short_term_memory(formatted_answer)
        self._create_long_term_memory(formatted_answer)
        self._create_external_memory(formatted_answer)
        return {"output": formatted_answer.output}

    def _populate_state_from_reasoning(self) -> None:
        """Populate agent state from the reasoning output if available."""
        try:
            # Check if the agent has reasoning output from the initial reasoning
            if hasattr(self.agent, '_last_reasoning_output') and self.agent._last_reasoning_output:
                reasoning_output = self.agent._last_reasoning_output

                # Extract structured plan if available
                if reasoning_output.plan.structured_plan:
                    self.agent_state.set_original_plan(reasoning_output.plan.structured_plan.steps)
                    self.agent_state.acceptance_criteria = reasoning_output.plan.structured_plan.acceptance_criteria
                elif reasoning_output.plan.plan:
                    # Fallback: try to extract steps from unstructured plan
                    plan_lines = [line.strip() for line in reasoning_output.plan.plan.split('\n') if line.strip()]
                    # Take meaningful lines that look like steps (skip headers, empty lines, etc.)
                    steps = []
                    for line in plan_lines:
                        if line and not line.startswith('###') and not line.startswith('**'):
                            steps.append(line)
                        if len(steps) >= 10:  # Limit to 10 steps
                            break
                    if steps:
                        self.agent_state.set_original_plan(steps)

                # Add state context to messages for coherence
                if self.agent_state.original_plan:
                    state_context = f"Initial plan loaded with {len(self.agent_state.original_plan)} steps."
                    self._append_message(state_context, role="assistant")

                # Clear the reasoning output to avoid using it again
                self.agent._last_reasoning_output = None

        except Exception as e:
            self._printer.print(
                content=f"Error populating state from reasoning: {str(e)}",
                color="yellow",
            )

    def _invoke_loop(self) -> AgentFinish:
        """
        Main loop to invoke the agent's thought process until it reaches a conclusion
        or the maximum number of iterations is reached.
        """
        formatted_answer = None
        while not isinstance(formatted_answer, AgentFinish):
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        formatted_answer,
                        printer=self._printer,
                        i18n=self._i18n,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                    )

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = get_llm_response(
                    llm=self.llm,
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=self._printer,
                )
                formatted_answer = process_llm_response(answer, self.use_stop_words)

                if isinstance(formatted_answer, AgentAction):
                    # Extract agent fingerprint if available
                    fingerprint_context = {}
                    if (
                        self.agent
                        and hasattr(self.agent, "security_config")
                        and hasattr(self.agent.security_config, "fingerprint")
                    ):
                        fingerprint_context = {
                            "agent_fingerprint": str(
                                self.agent.security_config.fingerprint
                            )
                        }

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
                    )
                    formatted_answer = self._handle_agent_action(
                        formatted_answer, tool_result
                    )

                    # Record detailed tool usage in agent state
                    if hasattr(formatted_answer, 'tool') and formatted_answer.tool:
                        # Extract tool arguments from the agent action
                        tool_args = {}
                        if hasattr(formatted_answer, 'tool_input') and formatted_answer.tool_input:
                            if isinstance(formatted_answer.tool_input, dict):
                                tool_args = formatted_answer.tool_input
                            elif isinstance(formatted_answer.tool_input, str):
                                # Try to parse JSON if it's a string
                                try:
                                    import json
                                    tool_args = json.loads(formatted_answer.tool_input)
                                except (json.JSONDecodeError, TypeError):
                                    tool_args = {"input": formatted_answer.tool_input}

                        # Truncate result for summary
                        result_summary = None
                        if tool_result and hasattr(tool_result, 'result'):
                            result_str = str(tool_result.result)
                            result_summary = result_str[:200] + "..." if len(result_str) > 200 else result_str

                        # Record the tool usage with arguments
                        self.agent_state.record_tool_usage(
                            tool_name=formatted_answer.tool,
                            arguments=tool_args,
                            result_summary=result_summary
                        )

                # Increment steps in agent state
                self.agent_state.increment_steps()

                if self._should_trigger_reasoning():
                    self._handle_mid_execution_reasoning()
                else:
                    self.steps_since_reasoning += 1

                self._invoke_step_callback(formatted_answer)
                self._append_message(formatted_answer.text, role="assistant")

            except OutputParserException as e:
                formatted_answer = handle_output_parser_exception(
                    e=e,
                    messages=self.messages,
                    iterations=self.iterations,
                    log_error_after=self.log_error_after,
                    printer=self._printer,
                )

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    # Do not retry on litellm errors
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=self._printer,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        i18n=self._i18n,
                        task_description=getattr(self.task, "description", None),
                        expected_output=getattr(self.task, "expected_output", None),
                    )
                    continue
                else:
                    handle_unknown_error(self._printer, e)
                    raise e
            finally:
                self.iterations += 1

        # During the invoke loop, formatted_answer alternates between AgentAction
        # (when the agent is using tools) and eventually becomes AgentFinish
        # (when the agent reaches a final answer). This assertion confirms we've
        # reached a final answer and helps type checking understand this transition.
        assert isinstance(formatted_answer, AgentFinish)
        self._show_logs(formatted_answer)
        return formatted_answer

    def _handle_agent_action(
        self, formatted_answer: AgentAction, tool_result: ToolResult
    ) -> Union[AgentAction, AgentFinish]:
        """Handle the AgentAction, execute tools, and process the results."""
        # Special case for add_image_tool
        add_image_tool = self._i18n.tools("add_image")
        if (
            isinstance(add_image_tool, dict)
            and formatted_answer.tool.casefold().strip()
            == add_image_tool.get("name", "").casefold().strip()
        ):
            self.messages.append({"role": "assistant", "content": tool_result.result})
            return formatted_answer

        return handle_agent_action_core(
            formatted_answer=formatted_answer,
            tool_result=tool_result,
            messages=self.messages,
            step_callback=self.step_callback,
            show_logs=self._show_logs,
        )

    def _invoke_step_callback(self, formatted_answer) -> None:
        """Invoke the step callback if it exists."""
        if self.step_callback:
            self.step_callback(formatted_answer)

    def _append_message(self, text: str, role: str = "assistant") -> None:
        """Append a message to the message list with the given role."""
        self.messages.append(format_message_for_llm(text, role=role))

    def _show_start_logs(self):
        """Show logs for the start of agent execution."""
        if self.agent is None:
            raise ValueError("Agent cannot be None")
        show_agent_logs(
            printer=self._printer,
            agent_role=self.agent.role,
            task_description=(
                getattr(self.task, "description") if self.task else "Not Found"
            ),
            verbose=self.agent.verbose
            or (hasattr(self, "crew") and getattr(self.crew, "verbose", False)),
        )

    def _show_logs(self, formatted_answer: Union[AgentAction, AgentFinish]):
        """Show logs for the agent's execution."""
        if self.agent is None:
            raise ValueError("Agent cannot be None")
        show_agent_logs(
            printer=self._printer,
            agent_role=self.agent.role,
            formatted_answer=formatted_answer,
            verbose=self.agent.verbose
            or (hasattr(self, "crew") and getattr(self.crew, "verbose", False)),
        )

    def _handle_crew_training_output(
        self, result: AgentFinish, human_feedback: Optional[str] = None
    ) -> None:
        """Handle the process of saving training data."""
        agent_id = str(self.agent.id)  # type: ignore
        train_iteration = (
            getattr(self.crew, "_train_iteration", None) if self.crew else None
        )

        if train_iteration is None or not isinstance(train_iteration, int):
            self._printer.print(
                content="Invalid or missing train iteration. Cannot save training data.",
                color="red",
            )
            return

        training_handler = CrewTrainingHandler(TRAINING_DATA_FILE)
        training_data = training_handler.load() or {}

        # Initialize or retrieve agent's training data
        agent_training_data = training_data.get(agent_id, {})

        if human_feedback is not None:
            # Save initial output and human feedback
            agent_training_data[train_iteration] = {
                "initial_output": result.output,
                "human_feedback": human_feedback,
            }
        else:
            # Save improved output
            if train_iteration in agent_training_data:
                agent_training_data[train_iteration]["improved_output"] = result.output
            else:
                self._printer.print(
                    content=(
                        f"No existing training data for agent {agent_id} and iteration "
                        f"{train_iteration}. Cannot save improved output."
                    ),
                    color="red",
                )
                return

        # Update the training data and save
        training_data[agent_id] = agent_training_data
        training_handler.save(training_data)

    def _format_prompt(self, prompt: str, inputs: Dict[str, str]) -> str:
        prompt = prompt.replace("{input}", inputs["input"])
        prompt = prompt.replace("{tool_names}", inputs["tool_names"])
        prompt = prompt.replace("{tools}", inputs["tools"])
        return prompt

    def _handle_human_feedback(self, formatted_answer: AgentFinish) -> AgentFinish:
        """Handle human feedback with different flows for training vs regular use.

        Args:
            formatted_answer: The initial AgentFinish result to get feedback on

        Returns:
            AgentFinish: The final answer after processing feedback
        """
        human_feedback = self._ask_human_input(formatted_answer.output)

        if self._is_training_mode():
            return self._handle_training_feedback(formatted_answer, human_feedback)

        return self._handle_regular_feedback(formatted_answer, human_feedback)

    def _is_training_mode(self) -> bool:
        """Check if crew is in training mode."""
        return bool(self.crew and self.crew._train)

    def _handle_training_feedback(
        self, initial_answer: AgentFinish, feedback: str
    ) -> AgentFinish:
        """Process feedback for training scenarios with single iteration."""
        self._handle_crew_training_output(initial_answer, feedback)
        self.messages.append(
            format_message_for_llm(
                self._i18n.slice("feedback_instructions").format(feedback=feedback)
            )
        )
        improved_answer = self._invoke_loop()
        self._handle_crew_training_output(improved_answer)
        self.ask_for_human_input = False
        return improved_answer

    def _handle_regular_feedback(
        self, current_answer: AgentFinish, initial_feedback: str
    ) -> AgentFinish:
        """Process feedback for regular use with potential multiple iterations."""
        feedback = initial_feedback
        answer = current_answer

        while self.ask_for_human_input:
            # If the user provides a blank response, assume they are happy with the result
            if feedback.strip() == "":
                self.ask_for_human_input = False
            else:
                answer = self._process_feedback_iteration(feedback)
                feedback = self._ask_human_input(answer.output)

        return answer

    def _process_feedback_iteration(self, feedback: str) -> AgentFinish:
        """Process a single feedback iteration."""
        self.messages.append(
            format_message_for_llm(
                self._i18n.slice("feedback_instructions").format(feedback=feedback)
            )
        )
        return self._invoke_loop()

    def _log_feedback_error(self, retry_count: int, error: Exception) -> None:
        """Log feedback processing errors."""
        self._printer.print(
            content=(
                f"Error processing feedback: {error}. "
                f"Retrying... ({retry_count + 1}/{MAX_LLM_RETRY})"
            ),
            color="red",
        )

    def _log_max_retries_exceeded(self) -> None:
        """Log when max retries for feedback processing are exceeded."""
        self._printer.print(
            content=(
                f"Failed to process feedback after {MAX_LLM_RETRY} attempts. "
                "Ending feedback loop."
            ),
            color="red",
        )

    def _should_trigger_reasoning(self) -> bool:
        """
        Determine if mid-execution reasoning should be triggered.

        Returns:
            bool: True if reasoning should be triggered, False otherwise.
        """
        if self.iterations == 0:
            return False

        if not hasattr(self.agent, "reasoning") or not self.agent.reasoning:
            return False

        if hasattr(self.agent, "reasoning_interval") and self.agent.reasoning_interval is not None:
            return self.steps_since_reasoning >= self.agent.reasoning_interval

        if hasattr(self.agent, "adaptive_reasoning") and self.agent.adaptive_reasoning:
            return self._should_adaptive_reason()

        return False

    def _should_adaptive_reason(self) -> bool:
        """
        Determine if adaptive reasoning should be triggered using LLM decision.
        Fallback to error detection if LLM decision fails.

        Returns:
            bool: True if adaptive reasoning should be triggered, False otherwise.
        """
        if self._has_recent_errors():
            try:
                from crewai.utilities.events.reasoning_events import AgentAdaptiveReasoningDecisionEvent
                from crewai.utilities.events.crewai_event_bus import crewai_event_bus

                crewai_event_bus.emit(
                    self.agent,
                    AgentAdaptiveReasoningDecisionEvent(
                        agent_role=self.agent.role,
                        task_id=str(self.task.id),
                        should_reason=True,
                        reasoning="Recent error indicators detected in previous messages.",
                    ),
                )
            except Exception:
                pass
            return True

        try:
            from crewai.utilities.reasoning_handler import AgentReasoning
            from crewai.agent import Agent

            current_progress = self._summarize_current_progress()

            # Build detailed tools used list from agent state
            tools_used_detailed = []
            for usage in self.agent_state.tool_usage_history:
                tool_desc = f"{usage.tool_name}"
                if usage.arguments:
                    args_preview = ", ".join(f"{k}={v}" for k, v in list(usage.arguments.items())[:2])
                    tool_desc += f"({args_preview})"
                tools_used_detailed.append(tool_desc)

            # Get tool usage statistics and patterns
            tool_stats = self.agent_state.get_tools_summary()

            # Detect patterns in tool usage
            tool_patterns = self._detect_tool_patterns()
            if tool_patterns:
                tool_stats['recent_patterns'] = tool_patterns

            reasoning_handler = AgentReasoning(task=self.task, agent=cast(Agent, self.agent))

            return reasoning_handler.should_adaptive_reason_llm(
                current_steps=self.iterations,
                tools_used=tools_used_detailed,
                current_progress=current_progress,
                tool_usage_stats=tool_stats
            )
        except Exception as e:
            self._printer.print(
                content=f"Error during adaptive reasoning decision: {str(e)}. Using fallback error detection.",
                color="yellow",
            )
            return False

    def _detect_tool_patterns(self) -> Optional[str]:
        """
        Detect patterns in recent tool usage that might indicate issues.

        Returns:
            Optional[str]: Description of detected patterns, or None
        """
        if not self.agent_state.tool_usage_history:
            return None

        patterns = []

        # Check for repeated use of the same tool with similar arguments
        recent_tools = self.agent_state.tool_usage_history[-5:] if len(self.agent_state.tool_usage_history) >= 5 else self.agent_state.tool_usage_history

        # Count consecutive uses of the same tool
        if len(recent_tools) >= 2:
            consecutive_count = 1
            for i in range(1, len(recent_tools)):
                if recent_tools[i].tool_name == recent_tools[i-1].tool_name:
                    consecutive_count += 1
                    if consecutive_count >= 3:
                        patterns.append(f"Same tool ({recent_tools[i].tool_name}) used {consecutive_count} times consecutively")
                else:
                    consecutive_count = 1

        # Check for tools with empty or error results
        error_count = 0
        for usage in recent_tools:
            if usage.result_summary and any(keyword in usage.result_summary.lower()
                                           for keyword in ['error', 'failed', 'not found', 'empty']):
                error_count += 1

        if error_count >= 2:
            patterns.append(f"{error_count} tools returned errors or empty results recently")

        # Check for rapid tool switching (might indicate confusion)
        if len(set(usage.tool_name for usage in recent_tools)) == len(recent_tools) and len(recent_tools) >= 4:
            patterns.append("Rapid switching between different tools without repetition")

        return "; ".join(patterns) if patterns else None

    def _handle_mid_execution_reasoning(self) -> None:
        """
        Handle mid-execution reasoning by calling the reasoning handler.
        """
        if not hasattr(self.agent, "reasoning") or not self.agent.reasoning:
            return

        try:
            from crewai.utilities.reasoning_handler import AgentReasoning

            current_progress = self._summarize_current_progress()

            # Include agent state in progress summary
            state_info = f"\n\n{self.agent_state.to_context_string()}"
            current_progress += state_info

            from crewai.agent import Agent

            reasoning_handler = AgentReasoning(task=self.task, agent=cast(Agent, self.agent))

            # Build detailed tools used list from agent state
            tools_used_detailed = []
            for usage in self.agent_state.tool_usage_history:
                tool_desc = f"{usage.tool_name}"
                if usage.arguments:
                    args_preview = ", ".join(f"{k}={v}" for k, v in list(usage.arguments.items())[:2])
                    tool_desc += f"({args_preview})"
                tools_used_detailed.append(tool_desc)

            reasoning_output = reasoning_handler.handle_mid_execution_reasoning(
                current_steps=self.iterations,
                tools_used=tools_used_detailed,
                current_progress=current_progress,
                iteration_messages=self.messages
            )

            # Update agent state with new plan if available
            if reasoning_output.plan.structured_plan:
                self.agent_state.update_last_plan(reasoning_output.plan.structured_plan.steps)
                # Update acceptance criteria if they changed
                if reasoning_output.plan.structured_plan.acceptance_criteria:
                    self.agent_state.acceptance_criteria = reasoning_output.plan.structured_plan.acceptance_criteria

            # Add a note about the reasoning update to scratchpad
            self.agent_state.add_to_scratchpad(
                f"reasoning_update_{self.iterations}",
                {
                    "reason": "Mid-execution reasoning triggered",
                    "updated_plan": bool(reasoning_output.plan.structured_plan)
                }
            )

            updated_plan_msg = (
                self._i18n.retrieve("reasoning", "mid_execution_reasoning_update").format(
                    plan=reasoning_output.plan.plan
                ) +
                f"\n\nUpdated State:\n{self.agent_state.to_context_string()}" +
                "\n\nRemember: strictly follow the updated plan above and ensure the final answer fully meets the EXPECTED OUTPUT criteria."
            )

            self._append_message(updated_plan_msg, role="assistant")

            self.steps_since_reasoning = 0

        except Exception as e:
            self._printer.print(
                content=f"Error during mid-execution reasoning: {str(e)}",
                color="red",
            )

    def _summarize_current_progress(self) -> str:
        """
        Create a summary of the current execution progress.

        Returns:
            str: A summary of the current progress.
        """
        recent_messages = self.messages[-5:] if len(self.messages) >= 5 else self.messages

        summary = f"After {self.iterations} steps, "

        # Use tool usage history from agent state for better context
        if self.agent_state.tool_usage_history:
            tool_summary = self.agent_state.get_tools_summary()
            summary += f"I've used {tool_summary['total_tool_uses']} tools ({tool_summary['unique_tools']} unique). "

            # Include most frequently used tools
            if tool_summary['tools_by_frequency']:
                top_tools = list(tool_summary['tools_by_frequency'].items())[:3]
                tools_str = ", ".join(f"{tool} ({count}x)" for tool, count in top_tools)
                summary += f"Most used: {tools_str}. "

            # Include details of the last tool use
            if self.agent_state.tool_usage_history:
                last_tool = self.agent_state.tool_usage_history[-1]
                summary += f"Last tool: {last_tool.tool_name}"
                if last_tool.arguments:
                    args_str = ", ".join(f"{k}={v}" for k, v in list(last_tool.arguments.items())[:2])
                    summary += f" with args ({args_str})"
                summary += ". "
        else:
            summary += "I haven't used any tools yet. "

        if recent_messages:
            last_message = recent_messages[-1].get("content", "")
            if len(last_message) > 100:
                last_message = last_message[:100] + "..."
            summary += f"Most recent action: {last_message}"

        return summary

    def _has_recent_errors(self) -> bool:
        """Check for error indicators in recent messages."""
        error_indicators = ["error", "exception", "failed", "unable to", "couldn't"]
        recent_messages = self.messages[-3:] if len(self.messages) >= 3 else self.messages

        for message in recent_messages:
            content = message.get("content", "").lower()
            if any(indicator in content for indicator in error_indicators):
                return True
        return False
