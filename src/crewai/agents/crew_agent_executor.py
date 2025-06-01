from typing import Any, Callable, Dict, List, Optional, Union, cast, Tuple
import json

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from crewai.agents.agent_state import AgentState
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from crewai.agents.tools_handler import ToolsHandler
from crewai.tools.agent_tools.scratchpad_tool import ScratchpadTool
from crewai.llm import BaseLLM
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_types import ToolResult
from crewai.utilities import I18N, Printer
from crewai.utilities.agent_utils import (
    enforce_rpm_limit,
    format_message_for_llm,
    get_llm_response,
    get_tool_names,
    handle_agent_action_core,
    handle_context_length,
    handle_max_iterations_exceeded,
    handle_output_parser_exception,
    handle_unknown_error,
    has_reached_max_iterations,
    is_context_length_exceeded,
    process_llm_response,
    render_text_description_and_args,
    show_agent_logs,
    parse_tools,
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
        self.agent_state: AgentState = AgentState(
            task_id=str(task.id) if task else None
        )
        self.scratchpad_tool: Optional[ScratchpadTool] = None
        existing_stop = self.llm.stop or []
        self.llm.stop = list(
            set(
                existing_stop + self.stop
                if isinstance(existing_stop, list)
                else self.stop
            )
        )

        # Initialize scratchpad tool if reasoning is enabled
        if hasattr(self.agent, "reasoning") and self.agent.reasoning:
            self._initialize_scratchpad_tool()

    def _initialize_scratchpad_tool(self) -> None:
        """Initialize the scratchpad tool and add it to available tools."""
        self.scratchpad_tool = ScratchpadTool(scratchpad_data=self.agent_state.scratchpad)

        # Add to tools list if not already present
        tool_names = [tool.name for tool in self.tools]
        if self.scratchpad_tool.name not in tool_names:
            # Use parse_tools to convert to CrewStructuredTool
            parsed_scratchpad_tools = parse_tools([self.scratchpad_tool])
            if parsed_scratchpad_tools:
                structured_scratchpad_tool = parsed_scratchpad_tools[0]
                self.tools.append(structured_scratchpad_tool)

                # Update tool mappings
                self.tool_name_to_tool_map[self.scratchpad_tool.name] = structured_scratchpad_tool

                # Update tools names and descriptions
                self.tools_names = get_tool_names(self.tools)
                self.tools_description = render_text_description_and_args(self.tools)

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

        self._create_short_term_memory(formatted_answer)
        self._create_long_term_memory(formatted_answer)
        self._create_external_memory(formatted_answer)
        return {"output": formatted_answer.output}

    def _populate_state_from_reasoning(self) -> None:
        """Populate agent state from the reasoning output if available."""
        try:
            # Check if the agent has reasoning output from the initial reasoning
            if (
                hasattr(self.agent, "_last_reasoning_output")
                and self.agent._last_reasoning_output
            ):
                reasoning_output = self.agent._last_reasoning_output

                # Extract structured plan if available
                if reasoning_output.plan.structured_plan:
                    self.agent_state.set_original_plan(
                        reasoning_output.plan.structured_plan.steps
                    )
                    self.agent_state.acceptance_criteria = (
                        reasoning_output.plan.structured_plan.acceptance_criteria
                    )
                elif reasoning_output.plan.plan:
                    # Fallback: try to extract steps from unstructured plan
                    plan_lines = [
                        line.strip()
                        for line in reasoning_output.plan.plan.split("\n")
                        if line.strip()
                    ]
                    # Take meaningful lines that look like steps (skip headers, empty lines, etc.)
                    steps = []
                    for line in plan_lines:
                        if (
                            line
                            and not line.startswith("###")
                            and not line.startswith("**")
                        ):
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
                print(f"\n[DEBUG] Starting iteration {self.iterations + 1}, max_iter: {self.max_iter}")

                if has_reached_max_iterations(self.iterations, self.max_iter):
                    print(f"[DEBUG] Max iterations reached")
                    formatted_answer = handle_max_iterations_exceeded(
                        formatted_answer,
                        printer=self._printer,
                        i18n=self._i18n,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                    )

                enforce_rpm_limit(self.request_within_rpm_limit)

                print(f"[DEBUG] About to call LLM with {len(self.messages)} messages")
                answer = get_llm_response(
                    llm=self.llm,
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=self._printer,
                )
                print(f"[DEBUG] LLM response received: {answer[:100]}..." if answer else "[DEBUG] No LLM response")

                formatted_answer = process_llm_response(answer, self.use_stop_words)
                print(f"[DEBUG] Formatted answer type: {type(formatted_answer).__name__}")

                # Check if agent is trying to finish but hasn't met criteria
                if isinstance(formatted_answer, AgentFinish):
                    print(f"[DEBUG] Agent trying to finish - checking acceptance criteria")
                    # Validate acceptance criteria if reasoning is enabled and criteria exist
                    if (hasattr(self.agent, "reasoning") and self.agent.reasoning
                        and self.agent_state.acceptance_criteria):

                        self._printer.print(
                            content="\nValidating acceptance criteria before finalizing...",
                            color="cyan"
                        )

                        print(f"[DEBUG] Starting validation of {len(self.agent_state.acceptance_criteria)} criteria")
                        is_valid, unmet_criteria = self._validate_acceptance_criteria(formatted_answer.output)
                        print(f"[DEBUG] Validation result: is_valid={is_valid}, unmet={len(unmet_criteria)}")

                        if not is_valid:
                            # Prevent task completion and force retry
                            self._printer.print(
                                content=f"\n❌ Cannot finalize - {len(unmet_criteria)} acceptance criteria not met:",
                                color="red"
                            )
                            for criterion in unmet_criteria:
                                self._printer.print(
                                    content=f"   • {criterion}",
                                    color="yellow"
                                )

                            # Create retry prompt
                            print(f"[DEBUG] Creating criteria retry prompt")
                            retry_prompt = self._create_criteria_retry_prompt(unmet_criteria)

                            # Add retry prompt to messages
                            self._append_message(retry_prompt, role="user")

                            # Force another iteration by resetting formatted_answer
                            formatted_answer = None
                            print(f"[DEBUG] Forcing another iteration due to unmet criteria")

                            # Continue the loop
                            continue
                        else:
                            self._printer.print(
                                content="\n✅ All acceptance criteria met!",
                                color="green"
                            )

                if isinstance(formatted_answer, AgentAction):
                    print(f"[DEBUG] Agent action: tool={formatted_answer.tool}")
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

                    print(f"[DEBUG] Executing tool: {formatted_answer.tool}")
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
                    print(f"[DEBUG] Tool execution completed")

                    formatted_answer = self._handle_agent_action(
                        formatted_answer, tool_result
                    )

                    # Record detailed tool usage in agent state
                    if hasattr(formatted_answer, "tool") and formatted_answer.tool:
                        # Extract tool arguments from the agent action
                        tool_args = {}
                        if (
                            hasattr(formatted_answer, "tool_input")
                            and formatted_answer.tool_input
                        ):
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
                        if tool_result and hasattr(tool_result, "result"):
                            result_str = str(tool_result.result)
                            result_summary = (
                                result_str[:200] + "..."
                                if len(result_str) > 200
                                else result_str
                            )

                        # Record the tool usage with arguments
                        self.agent_state.record_tool_usage(
                            tool_name=formatted_answer.tool,
                            arguments=tool_args,
                            result_summary=result_summary,
                        )

                        # Extract relevant information to scratchpad if reasoning is enabled
                        # Note: This only happens for actual tool executions (AgentAction with tool),
                        # not for reasoning steps or other agent outputs
                        if (
                            hasattr(self.agent, "reasoning")
                            and self.agent.reasoning
                            and tool_result
                            and formatted_answer.tool != "Access Scratchpad Memory"  # Skip scratchpad tool itself
                            and self._is_tool_execution_successful(tool_result)  # Only for successful executions
                        ):
                            print(f"[DEBUG] Starting scratchpad extraction for {formatted_answer.tool}")
                            self._extract_tool_result_to_scratchpad(
                                tool_name=formatted_answer.tool,
                                tool_args=tool_args,
                                tool_result=tool_result,
                            )

                            # Always print agent state (temporary debug)
                            print(
                                f"Agent State:\n"
                                f"Raw: {self.agent_state.model_dump_json()}\n"
                                f"[AGENT STATE] Step {self.agent_state.steps_completed}:\n"
                                f"Original Plan: {getattr(self.agent_state, 'original_plan', None)}\n"
                                f"Tool: {formatted_answer.tool}\n"
                                f"Scratchpad: {self.agent_state.scratchpad}\n"
                                f"Tool History: {len(self.agent_state.tool_usage_history)} entries\n"
                                f"Full State Context:\n{self.agent_state.to_context_string()}"
                            )

                # Increment steps in agent state
                self.agent_state.increment_steps()

                # Update scratchpad tool if it exists
                if self.scratchpad_tool and self.agent_state.scratchpad:
                    print(f"[DEBUG] Updating scratchpad tool")
                    self._update_scratchpad_tool()

                if self._should_trigger_reasoning():
                    print(f"[DEBUG] Triggering mid-execution reasoning")
                    self._handle_mid_execution_reasoning()
                    print(f"[DEBUG] Mid-execution reasoning completed")
                else:
                    self.steps_since_reasoning += 1

                self._invoke_step_callback(formatted_answer)
                self._append_message(formatted_answer.text, role="assistant")

            except OutputParserException as e:
                print(f"[DEBUG] OutputParserException: {str(e)}")
                formatted_answer = handle_output_parser_exception(
                    e=e,
                    messages=self.messages,
                    iterations=self.iterations,
                    log_error_after=self.log_error_after,
                    printer=self._printer,
                )

            except Exception as e:
                print(f"[DEBUG] Exception in invoke loop: {type(e).__name__}: {str(e)}")
                if e.__class__.__module__.startswith("litellm"):
                    # Do not retry on litellm errors
                    raise e
                if is_context_length_exceeded(e):
                    print(f"[DEBUG] Context length exceeded, handling...")
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
                print(f"[DEBUG] Iteration {self.iterations} completed")

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

        if (
            hasattr(self.agent, "reasoning_interval")
            and self.agent.reasoning_interval is not None
        ):
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
                from crewai.utilities.events.reasoning_events import (
                    AgentAdaptiveReasoningDecisionEvent,
                )
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
                    args_preview = ", ".join(
                        f"{k}={v}" for k, v in list(usage.arguments.items())[:2]
                    )
                    tool_desc += f"({args_preview})"
                tools_used_detailed.append(tool_desc)

            # Get tool usage statistics and patterns
            tool_stats = self.agent_state.get_tools_summary()

            # Detect patterns in tool usage
            tool_patterns = self._detect_tool_patterns()
            if tool_patterns:
                tool_stats["recent_patterns"] = tool_patterns

            reasoning_handler = AgentReasoning(
                task=self.task, agent=cast(Agent, self.agent)
            )

            return reasoning_handler.should_adaptive_reason_llm(
                current_steps=self.iterations,
                tools_used=tools_used_detailed,
                current_progress=current_progress,
                tool_usage_stats=tool_stats,
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
        recent_tools = (
            self.agent_state.tool_usage_history[-5:]
            if len(self.agent_state.tool_usage_history) >= 5
            else self.agent_state.tool_usage_history
        )

        # Count consecutive uses of the same tool
        if len(recent_tools) >= 2:
            consecutive_count = 1
            for i in range(1, len(recent_tools)):
                if recent_tools[i].tool_name == recent_tools[i - 1].tool_name:
                    consecutive_count += 1
                    if consecutive_count >= 3:
                        patterns.append(
                            f"Same tool ({recent_tools[i].tool_name}) used {consecutive_count} times consecutively"
                        )
                else:
                    consecutive_count = 1

        # Check for tools with empty or error results
        error_count = 0
        for usage in recent_tools:
            if usage.result_summary and any(
                keyword in usage.result_summary.lower()
                for keyword in ["error", "failed", "not found", "empty"]
            ):
                error_count += 1

        if error_count >= 2:
            patterns.append(
                f"{error_count} tools returned errors or empty results recently"
            )

        # Check for rapid tool switching (might indicate confusion)
        if (
            len(set(usage.tool_name for usage in recent_tools)) == len(recent_tools)
            and len(recent_tools) >= 4
        ):
            patterns.append(
                "Rapid switching between different tools without repetition"
            )

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

            reasoning_handler = AgentReasoning(
                task=self.task, agent=cast(Agent, self.agent)
            )

            # Build detailed tools used list from agent state
            tools_used_detailed = []
            for usage in self.agent_state.tool_usage_history:
                tool_desc = f"{usage.tool_name}"
                if usage.arguments:
                    args_preview = ", ".join(
                        f"{k}={v}" for k, v in list(usage.arguments.items())[:2]
                    )
                    tool_desc += f"({args_preview})"
                tools_used_detailed.append(tool_desc)

            reasoning_output = reasoning_handler.handle_mid_execution_reasoning(
                current_steps=self.iterations,
                tools_used=tools_used_detailed,
                current_progress=current_progress,
                iteration_messages=self.messages,
            )

            # Update acceptance criteria if they changed from the reasoning output
            if reasoning_output.plan.structured_plan:
                if reasoning_output.plan.structured_plan.acceptance_criteria:
                    self.agent_state.acceptance_criteria = (
                        reasoning_output.plan.structured_plan.acceptance_criteria
                    )

            # Don't add reasoning metadata to scratchpad - keep it exclusively for tool results

            updated_plan_msg = (
                self._i18n.retrieve(
                    "reasoning", "mid_execution_reasoning_update"
                ).format(plan=reasoning_output.plan.plan)
                + f"\n\nUpdated State:\n{self.agent_state.to_context_string()}"
                + "\n\nRemember: strictly follow the updated plan above and ensure the final answer fully meets the EXPECTED OUTPUT criteria."
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
        recent_messages = (
            self.messages[-5:] if len(self.messages) >= 5 else self.messages
        )

        summary = f"After {self.iterations} steps, "

        # Use tool usage history from agent state for better context
        if self.agent_state.tool_usage_history:
            tool_summary = self.agent_state.get_tools_summary()
            summary += f"I've used {tool_summary['total_tool_uses']} tools ({tool_summary['unique_tools']} unique). "

            # Include most frequently used tools
            if tool_summary["tools_by_frequency"]:
                top_tools = list(tool_summary["tools_by_frequency"].items())[:3]
                tools_str = ", ".join(f"{tool} ({count}x)" for tool, count in top_tools)
                summary += f"Most used: {tools_str}. "

            # Include details of the last tool use
            if self.agent_state.tool_usage_history:
                last_tool = self.agent_state.tool_usage_history[-1]
                summary += f"Last tool: {last_tool.tool_name}"
                if last_tool.arguments:
                    args_str = ", ".join(
                        f"{k}={v}" for k, v in list(last_tool.arguments.items())[:2]
                    )
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
        recent_messages = (
            self.messages[-3:] if len(self.messages) >= 3 else self.messages
        )

        for message in recent_messages:
            content = message.get("content", "").lower()
            if any(indicator in content for indicator in error_indicators):
                return True
        return False

    def _extract_tool_result_to_scratchpad(
        self, tool_name: str, tool_args: Dict[str, Any], tool_result: ToolResult
    ) -> None:
        """Extract relevant information from tool result using LLM and add to scratchpad.

        This method uses the agent's LLM to intelligently extract and summarize
        important information from tool results, storing it in the agent's scratchpad
        for future reference during task execution.

        Args:
            tool_name: Name of the tool that was executed
            tool_args: Arguments that were passed to the tool
            tool_result: The result returned by the tool
        """
        print(f"[DEBUG] _extract_tool_result_to_scratchpad started for tool: {tool_name}")
        try:
            # Check result size and potentially skip LLM extraction for very large results
            result_str = str(tool_result.result)
            result_size = len(result_str)
            print(f"[DEBUG] Tool result size: {result_size} characters")

            # For very large results (>100KB), skip LLM extraction and store directly
            if result_size > 100000:
                print(f"[DEBUG] Result too large ({result_size} chars), storing directly without LLM extraction")
                scratchpad_key = tool_name.replace("_", "")

                # Try to parse as JSON if possible
                try:
                    if isinstance(tool_result.result, str):
                        result_data = json.loads(tool_result.result)
                    else:
                        result_data = tool_result.result
                except:
                    result_data = tool_result.result

                self.agent_state.add_to_scratchpad(
                    scratchpad_key,
                    {
                        "data": result_data,
                        "tool": tool_name,
                        "tool_args": tool_args,
                        "large_result": True,
                        "size": result_size
                    }
                )
                print(f"[DEBUG] Large result stored directly to scratchpad")
                return

            # Create a prompt for the LLM to extract relevant information
            result_preview = str(tool_result.result)[:200] + "..." if len(str(tool_result.result)) > 200 else str(tool_result.result)
            print(f"[DEBUG] Tool result preview: {result_preview}")

            extraction_prompt = f"""Given the following tool execution result, extract and summarize the most relevant information that would be useful for completing the current task.

Tool Name: {tool_name}
Tool Arguments: {json.dumps(tool_args, indent=2) if tool_args else "None"}
Tool Result: {tool_result.result}

Current Task Context:
- Task Description: {self.task.description if self.task else "Not specified"}
- Expected Output: {self.task.expected_output if self.task else "Not specified"}
- Steps Completed: {self.agent_state.steps_completed}

Instructions:
1. Identify the KEY INFORMATION from the tool result that directly relates to the task
2. Extract any important data points, facts, or findings
3. Note any errors, warnings, or issues that might affect task completion
4. Summarize in a concise format (max 3-5 bullet points)
5. Focus on information that will be useful for subsequent steps
6. Generate a descriptive key name that explains what data is being stored (e.g., "email_and_thread_ids", "search_results", "file_contents", etc.)
7. IMPORTANT: When extracting data_points, include ALL items from lists or collections, do not truncate or summarize the data

Respond in the following JSON format:
{{
    "suggested_key_name": "descriptive_name_for_this_data",
    "key_findings": ["finding1", "finding2", ...],
    "data_points": {{"key": "value", ...}} or [list of items],
    "issues": ["issue1", "issue2", ...] or null if none,
    "relevance_score": 1-10 (how relevant this result is to the task)
}}

Note: For data_points, preserve the complete data structure. If it's a list of items (like email IDs, search results, etc.), include ALL items."""

            # Create messages for LLM call
            messages = [format_message_for_llm(extraction_prompt, role="user")]

            # Call LLM to extract information
            try:
                print(f"[DEBUG] Calling LLM for scratchpad extraction...")
                extraction_response = get_llm_response(
                    llm=self.llm,
                    messages=messages,
                    callbacks=self.callbacks,
                    printer=self._printer,
                )
                print(f"[DEBUG] LLM extraction response received, length: {len(extraction_response)}")

                # Try to parse the JSON response directly
                try:
                    extracted_info = json.loads(extraction_response)
                    print(f"[DEBUG] Successfully parsed JSON directly")
                except json.JSONDecodeError:
                    print(f"[DEBUG] Failed to parse JSON directly, trying to extract from markdown...")
                    # If direct parsing fails, try to extract JSON from the response
                    # The LLM might have wrapped it in markdown code blocks or added extra text
                    json_match = None

                    # Try to find JSON in markdown code blocks
                    import re

                    json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
                    matches = re.findall(json_pattern, extraction_response, re.DOTALL)

                    if matches:
                        print(f"[DEBUG] Found {len(matches)} JSON blocks in markdown")
                        # Try to parse the first match
                        for match in matches:
                            try:
                                json_match = json.loads(match)
                                print(f"[DEBUG] Successfully parsed JSON from markdown")
                                break
                            except json.JSONDecodeError:
                                continue

                    # If no markdown JSON found, try to find raw JSON object
                    if not json_match:
                        print(f"[DEBUG] No markdown JSON found, looking for raw JSON...")
                        # Look for JSON object in the response
                        json_start = extraction_response.find("{")
                        json_end = extraction_response.rfind("}")
                        if (
                            json_start != -1
                            and json_end != -1
                            and json_end > json_start
                        ):
                            try:
                                potential_json = extraction_response[
                                    json_start : json_end + 1
                                ]
                                json_match = json.loads(potential_json)
                                print(f"[DEBUG] Successfully extracted raw JSON")
                            except json.JSONDecodeError:
                                print(f"[DEBUG] Failed to parse raw JSON")
                                pass

                    if json_match:
                        extracted_info = json_match
                    else:
                        # Couldn't parse JSON, raise to trigger fallback
                        print(f"[DEBUG] Could not extract any valid JSON, triggering fallback")
                        raise json.JSONDecodeError(
                            "Could not extract JSON", extraction_response, 0
                        )

                # Process the extracted info
                # Use the suggested key name or fall back to default
                suggested_key = extracted_info.get("suggested_key_name", "")
                if suggested_key and suggested_key.replace("_", "").isalnum():
                    scratchpad_key = suggested_key
                else:
                    # Generate a meaningful key from tool name
                    scratchpad_key = tool_name.replace("_", "")
                print(f"[DEBUG] Using scratchpad key: {scratchpad_key}")

                # Get the data points
                data_points = extracted_info.get("data_points", {})

                # Simplify the data structure based on what's extracted
                if isinstance(data_points, list):
                    # If data_points is already a list, store it directly
                    data_to_store = data_points
                elif isinstance(data_points, dict) and len(data_points) == 1:
                    # If it's a dict with a single key containing a list, extract the list
                    single_key = list(data_points.keys())[0]
                    if isinstance(data_points[single_key], list):
                        data_to_store = data_points[single_key]
                    else:
                        data_to_store = data_points
                else:
                    data_to_store = data_points

                # Store based on relevance score
                relevance_score = extracted_info.get("relevance_score", 0)
                print(f"[DEBUG] Relevance score: {relevance_score}")

                if relevance_score >= 7:
                    # For high relevance, store just the data
                    self.agent_state.add_to_scratchpad(scratchpad_key, data_to_store)
                    print(f"[DEBUG] Stored high relevance data to scratchpad")
                else:
                    # For lower relevance, include more context
                    self.agent_state.add_to_scratchpad(
                        scratchpad_key,
                        {
                            "data": data_to_store,
                            "tool": tool_name,
                            "findings": extracted_info.get("key_findings", []),
                            "relevance": relevance_score,
                        },
                    )
                    print(f"[DEBUG] Stored lower relevance data with context to scratchpad")

                # Also store key findings if present and relevance is high
                if relevance_score >= 7 and extracted_info.get("key_findings"):
                    current_findings = self.agent_state.scratchpad.get(
                        "key_findings", []
                    )
                    current_findings.extend(extracted_info["key_findings"])
                    self.agent_state.add_to_scratchpad(
                        "key_findings", current_findings[-10:]
                    )
                    print(f"[DEBUG] Updated key findings in scratchpad")

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"[DEBUG] Exception during extraction: {type(e).__name__}: {str(e)}")
                # Fallback for when we can't extract structured data
                # Try to generate a meaningful key name from tool name
                scratchpad_key = tool_name.replace("_", "")

                # Store the complete result without truncation
                self.agent_state.add_to_scratchpad(
                    scratchpad_key,
                    {
                        "raw_response": extraction_response,  # Store complete response
                        "tool_result": tool_result.result,  # Store complete result
                        "extraction_failed": True,
                        "tool_args": tool_args
                    },
                )
                print(f"[DEBUG] Stored fallback data to scratchpad")

        except Exception as e:
            # Log error but don't fail the entire execution
            print(f"[DEBUG] Failed to extract tool result: {type(e).__name__}: {str(e)}")
            self._printer.print(
                content=f"Failed to extract tool result to scratchpad: {str(e)}",
                color="yellow",
            )
            # Still store complete information even if extraction fails
            fallback_key = f"{tool_name}_raw_{self.agent_state.steps_completed}"
            self.agent_state.add_to_scratchpad(
                fallback_key,
                {
                    "error": f"Extraction failed: {str(e)}",
                    "tool_result": tool_result.result,  # Store complete result
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "raw_data": True
                },
            )
            print(f"[DEBUG] Stored error fallback data to scratchpad")

        print(f"[DEBUG] _extract_tool_result_to_scratchpad completed")

    def _update_scratchpad_tool(self) -> None:
        """Update the scratchpad tool with current state data."""
        if not self.scratchpad_tool:
            return

        # Update the tool's data
        self.scratchpad_tool.update_scratchpad(self.agent_state.scratchpad)

        # Find and update the tool in our tools list
        for i, tool in enumerate(self.tools):
            if hasattr(tool, 'name') and tool.name == self.scratchpad_tool.name:
                # Update the description on the existing tool reference
                if hasattr(tool, '_tool') and hasattr(tool._tool, 'description'):
                    tool._tool.description = self.scratchpad_tool.description
                elif hasattr(tool, 'description'):
                    tool.description = self.scratchpad_tool.description
                break

        # Regenerate tools description to reflect the updated tool
        self.tools_description = render_text_description_and_args(self.tools)

    def _validate_acceptance_criteria(self, output: str) -> Tuple[bool, List[str]]:
        """Validate if the output meets acceptance criteria.

        Args:
            output: The final output to validate

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of unmet criteria)
        """
        print(f"[DEBUG] _validate_acceptance_criteria started")
        if not self.agent_state.acceptance_criteria:
            # No criteria to validate
            print(f"[DEBUG] No acceptance criteria to validate")
            return True, []

        # Create a single prompt to check all criteria
        criteria_list = "\n".join(
            f"{i}. {criterion}"
            for i, criterion in enumerate(self.agent_state.acceptance_criteria, 1)
        )
        print(f"[DEBUG] Validating {len(self.agent_state.acceptance_criteria)} criteria")

        validation_prompt = f"""Given the following task output and acceptance criteria, identify which criteria have NOT been met.

Task Output:
{output}

Expected Output Description:
{self.task.expected_output if self.task else "Not specified"}

Acceptance Criteria:
{criteria_list}

For each criterion, determine if it has been met or not met in the output.
Respond with a JSON object where keys are criterion numbers (1, 2, 3, etc.) and values are:
- "MET" if the criterion is satisfied
- "NOT MET: <brief reason>" if the criterion is not satisfied

Example response format:
{{
    "1": "MET",
    "2": "NOT MET: Missing specific examples",
    "3": "MET"
}}
"""

        try:
            print(f"[DEBUG] Calling LLM for criteria validation...")
            response = self.llm.call([
                {"role": "user", "content": validation_prompt}
            ])
            print(f"[DEBUG] LLM validation response received")

            # Parse the response as JSON
            import json
            response_str = str(response).strip()

            # Try to extract JSON from the response
            json_start = response_str.find('{')
            json_end = response_str.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_str[json_start:json_end]
                validation_results = json.loads(json_str)
                print(f"[DEBUG] Successfully parsed validation JSON")
            else:
                # Fallback if JSON not found
                self._logger.log("warning", f"Could not parse validation response as JSON: {response_str}")
                print(f"[DEBUG] Failed to parse validation response as JSON")
                # Assume all criteria not met if we can't parse
                return False, self.agent_state.acceptance_criteria

            # Process results
            unmet_criteria = []
            for i, criterion in enumerate(self.agent_state.acceptance_criteria, 1):
                result = validation_results.get(str(i), "NOT MET")
                if isinstance(result, str) and result.upper().startswith("NOT MET"):
                    unmet_criteria.append(criterion)
                    self._printer.print(
                        content=f"✗ Criterion not met: {criterion}",
                        color="yellow"
                    )
                else:
                    self._printer.print(
                        content=f"✓ Criterion met: {criterion}",
                        color="green"
                    )

            print(f"[DEBUG] Validation complete: {len(unmet_criteria)} unmet criteria")
            return len(unmet_criteria) == 0, unmet_criteria

        except Exception as e:
            print(f"[DEBUG] Error validating criteria: {type(e).__name__}: {str(e)}")
            self._logger.log("warning", f"Error validating criteria: {str(e)}")
            # If we can't validate, assume all criteria are not met to be safe
            return False, self.agent_state.acceptance_criteria

    def _create_criteria_retry_prompt(self, unmet_criteria: List[str]) -> str:
        """Create a prompt to retry task with unmet criteria.

        Args:
            unmet_criteria: List of criteria that weren't met

        Returns:
            str: The retry prompt
        """
        # Get task context
        task_description = self.task.description if self.task else "Not specified"
        expected_output = self.task.expected_output if self.task else "Not specified"

        # Build information about what's in the scratchpad
        scratchpad_info = ""
        scratchpad_data_summary = ""
        if self.scratchpad_tool and self.agent_state.scratchpad:
            scratchpad_keys = list(self.agent_state.scratchpad.keys())
            scratchpad_info = f"""
📦 YOUR SCRATCHPAD CONTAINS DATA:
{chr(10).join(f"  • '{key}'" for key in scratchpad_keys)}

TO ACCESS THIS DATA: Use the "Access Scratchpad Memory" tool with the key name.
Example:
Action: Access Scratchpad Memory
Action Input: {{"key": "{scratchpad_keys[0] if scratchpad_keys else 'key_name'}"}}
"""
            # Add summary of what's in scratchpad
            for key in scratchpad_keys[:3]:  # Show first 3 keys as examples
                value = self.agent_state.scratchpad[key]
                if isinstance(value, list):
                    scratchpad_data_summary += f"\n  - '{key}': contains {len(value)} items"
                elif isinstance(value, dict):
                    scratchpad_data_summary += f"\n  - '{key}': contains data with {len(value)} fields"
                else:
                    scratchpad_data_summary += f"\n  - '{key}': contains stored data"

        # Analyze what's missing based on criteria
        missing_data_hints = []
        for criterion in unmet_criteria:
            criterion_lower = criterion.lower()
            if "every email" in criterion_lower or "all" in criterion_lower:
                missing_data_hints.append("You need to retrieve ALL emails, not just a summary")
            if "date" in criterion_lower or "time" in criterion_lower:
                missing_data_hints.append("Include complete date/time information for each record")
            if "subject" in criterion_lower or "sender" in criterion_lower or "recipients" in criterion_lower:
                missing_data_hints.append("Ensure all email metadata (subject, sender, recipients) is included")
            if "format" in criterion_lower or "list" in criterion_lower:
                missing_data_hints.append("Format the data properly as requested")
            if "summary" in criterion_lower or "concise" in criterion_lower:
                missing_data_hints.append("Include a concise summary/snippet for each email")

        # Get available tools (excluding scratchpad tool)
        available_tools = [tool for tool in self.tools_names.split(", ") if tool != "Access Scratchpad Memory"]
        tools_hint = f"\n🛠️ AVAILABLE TOOLS: {', '.join(available_tools)}" if available_tools else ""

        # Get progress summary
        progress_summary = f"""
📊 CURRENT PROGRESS:
- Steps completed: {self.agent_state.steps_completed}
- Tools used: {len(self.agent_state.tool_usage_history)} times"""

        if self.agent_state.tool_usage_history:
            recent_tools = self.agent_state.tool_usage_history[-3:]
            progress_summary += f"\n- Recent tools: {', '.join(t.tool_name for t in recent_tools)}"

        prompt = f"""❌ VALIDATION FAILED - YOU CANNOT PROVIDE A FINAL ANSWER YET!

Your output is INCOMPLETE and missing critical information.

🎯 ORIGINAL TASK:
{task_description}

📋 EXPECTED OUTPUT:
{expected_output}

❌ UNMET CRITERIA:
{chr(10).join(f"❌ {criterion}" for criterion in unmet_criteria)}

⚠️ CRITICAL: You MUST go back to using tools to gather the missing data!

DO NOT attempt another "Final Answer" until you have ALL required data.
{progress_summary}

🔧 REQUIRED ACTIONS:
1. STOP trying to provide a Final Answer
2. Switch to using Action/Action Input format
3. Use tools to gather the missing information
{scratchpad_info}

💡 WHAT YOU'RE MISSING:
{chr(10).join(f"• {hint}" for hint in missing_data_hints) if missing_data_hints else "• Review the criteria and gather all required data"}
{scratchpad_data_summary}

📋 YOUR NEXT STEP:
You MUST use the following format to continue:

Thought: I need to gather the missing data using tools
Action: [tool name]
Action Input: {{"parameter": "value"}}
{tools_hint}

⚠️ IMPORTANT REMINDERS:
- The task requires you to retrieve EVERY email, not just summaries
- You already have data in your scratchpad - ACCESS IT FIRST with "Access Scratchpad Memory"
- Each email needs: date, time, subject, sender, recipients, and content snippet
- Continue retrieving details for ALL emails until complete
- Only provide a Final Answer after you have gathered ALL required data

CONTINUE WITH TOOL USAGE NOW - DO NOT ATTEMPT ANOTHER FINAL ANSWER."""

        return prompt

    def _is_tool_execution_successful(self, tool_result: ToolResult) -> bool:
        """Check if a tool execution was successful based on the tool result."""
        if tool_result.result is None or tool_result.result == "":
            return False

        # Check for common error indicators in the result
        result_str = str(tool_result.result).lower()
        error_indicators = [
            "error",
            "exception",
            "failed",
            "unable to",
            "couldn't",
            "not found",
            "invalid",
            "wrong tool name",
            "don't exist",
            "tool usage exception",
            "moving on then",
            "has reached its usage limit"
        ]

        # If any error indicator is found in the result, consider it a failure
        for indicator in error_indicators:
            if indicator in result_str:
                return False

        return True
