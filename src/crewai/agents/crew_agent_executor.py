from typing import Any, Callable, Dict, List, Optional, Union, cast, Tuple
import re

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
        self.agent_state: AgentState = AgentState()
        self.scratchpad_tool: Optional[ScratchpadTool] = None
        self.max_iterations_exceeded = False
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
        self.agent_state.reset()
        self.max_iterations_exceeded = False  # Reset the flag for new execution

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
                    self.agent_state.plan = reasoning_output.plan.structured_plan.steps
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
                        self.agent_state.plan = steps

                # Initialize progress tracking for criteria
                if self.agent_state.acceptance_criteria:
                    self.agent_state.initialize_criteria_progress()

                    # Set initial focus
                    self.agent_state.set_focus_and_next_steps(
                        focus=f"Starting task execution to meet {len(self.agent_state.acceptance_criteria)} acceptance criteria",
                        next_steps=["Begin with the first step of the plan", "Use appropriate tools to gather required data"]
                    )

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
                    self.max_iterations_exceeded = True  # Set flag

                    # Add informative message about skipping validation
                    if self.agent_state.acceptance_criteria:
                        self._printer.print(
                            content="\n⚠️ Max iterations reached - forcing completion without acceptance criteria validation",
                            color="yellow"
                        )

                    # Directly create a final answer based on current progress
                    # Extract any existing data from scratchpad or messages
                    final_output = self._create_forced_final_answer()

                    formatted_answer = AgentFinish(
                        thought="Maximum iterations reached - compiling available results",
                        output=final_output,
                        text=final_output
                    )
                    break  # Exit the loop immediately with the forced answer

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
                        and self.agent_state.acceptance_criteria
                        and not self.max_iterations_exceeded):

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

                    # Special handling for scratchpad tool errors
                    if (formatted_answer.tool == "Access Scratchpad Memory"
                        and tool_result
                        and hasattr(tool_result, 'result')
                        and isinstance(tool_result.result, str)
                        and "❌ KEY NOT FOUND:" in tool_result.result):
                        # Extract available keys from the error message
                        error_lines = tool_result.result.split('\n')
                        keys_section_start = False
                        available_keys = []

                        for line in error_lines:
                            if "AVAILABLE KEYS IN SCRATCHPAD:" in line:
                                keys_section_start = True
                                continue
                            if keys_section_start and line.strip().startswith("- '"):
                                key_match = re.search(r"- '([^']+)'", line)
                                if key_match:
                                    available_keys.append(key_match.group(1))
                            elif keys_section_start and not line.strip().startswith("- "):
                                break

                        if available_keys:
                            system_msg = (
                                f"⚠️ SCRATCHPAD ACCESS ERROR - PAY ATTENTION!\n\n"
                                f"You tried to access a key that doesn't exist.\n"
                                f"HERE ARE THE CORRECT KEYS YOU MUST USE:\n"
                                f"{chr(10).join(f'  ✓ {key}' for key in available_keys)}\n\n"
                                f"NEXT ACTION: Use 'Access Scratchpad Memory' with one of the keys above.\n"
                                f"Example: Action Input: {{\"key\": \"{available_keys[0]}\"}}"
                            )
                            self._append_message(system_msg, role="system")

                    # Extract to scratchpad if reasoning is enabled and tool was successful
                    if (
                        hasattr(self.agent, "reasoning")
                        and self.agent.reasoning
                        and tool_result
                        and formatted_answer.tool != "Access Scratchpad Memory"
                        and self._is_tool_execution_successful(tool_result)
                    ):
                        print(f"[DEBUG] Starting scratchpad extraction for {formatted_answer.tool}")
                        self._extract_tool_result_to_scratchpad(
                            tool_name=formatted_answer.tool,
                            tool_args=getattr(formatted_answer, 'tool_input', {}),
                            tool_result=tool_result,
                        )

                # After each step, update progress tracking
                if hasattr(self.agent, "reasoning") and self.agent.reasoning and self.agent_state.acceptance_criteria:
                    self._update_progress_tracking()

                # Update scratchpad tool if it exists
                if self.scratchpad_tool and self.agent_state.scratchpad:
                    print(f"[DEBUG] Updating scratchpad tool")
                    self._update_scratchpad_tool()

                # Inject progress context for the next iteration
                if self.agent_state.acceptance_criteria:
                    progress_context = self.agent_state.get_progress_context()
                    self._append_message(progress_context, role="system")

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
        """Extract relevant information from tool result and add to scratchpad with progress logging."""
        print(f"[DEBUG] _extract_tool_result_to_scratchpad started for tool: {tool_name}")
        try:
            # Generate a meaningful key from tool name
            scratchpad_key = tool_name.lower().replace(" ", "_").replace("_tool", "")

            # Store the result data
            if hasattr(tool_result, 'result'):
                result_data = tool_result.result

                # Try to parse JSON if it's a string
                if isinstance(result_data, str):
                    try:
                        import json
                        parsed_data = json.loads(result_data)
                        result_data = parsed_data
                    except:
                        pass  # Keep as string if not parseable

                # Store in scratchpad
                self.agent_state.add_to_scratchpad(scratchpad_key, result_data)
                print(f"[DEBUG] Stored result to scratchpad with key: {scratchpad_key}")

                # Extract item count and IDs for logging
                items_processed = []
                if isinstance(result_data, list):
                    # Try to extract IDs from list items
                    for item in result_data[:10]:  # Limit to first 10 for logging
                        if isinstance(item, dict):
                            for id_field in ['id', 'ID', 'uid', 'uuid', 'message_id', 'email_id']:
                                if id_field in item:
                                    items_processed.append(str(item[id_field]))
                                    break

                    # Log progress with item details
                    self.agent_state.log_progress(
                        action=f"Executed {tool_name}",
                        result=f"Retrieved {len(result_data)} items and stored in scratchpad",
                        items_processed=items_processed[:10]  # Log up to 10 IDs
                    )

                    self._printer.print(
                        content=f"✓ Stored {len(result_data)} items from {tool_name} to scratchpad",
                        color="green"
                    )
                elif isinstance(result_data, dict):
                    # For dict results, log the action
                    self.agent_state.log_progress(
                        action=f"Executed {tool_name}",
                        result=f"Retrieved data object and stored in scratchpad"
                    )

                    self._printer.print(
                        content=f"✓ Stored data from {tool_name} to scratchpad",
                        color="green"
                    )
                else:
                    # For other types, just log
                    self.agent_state.log_progress(
                        action=f"Executed {tool_name}",
                        result=f"Stored result in scratchpad"
                    )

        except Exception as e:
            print(f"[DEBUG] Failed to extract tool result: {type(e).__name__}: {str(e)}")
            # Log error but don't fail the entire execution
            self._printer.print(
                content=f"Failed to extract tool result to scratchpad: {str(e)}",
                color="yellow"
            )

            self.agent_state.log_progress(
                action=f"Failed to extract {tool_name} result",
                result=str(e)
            )

    def _update_scratchpad_tool(self) -> None:
        """Update the scratchpad tool with current state data."""
        if not self.scratchpad_tool:
            return

        # Update the tool's data
        self.scratchpad_tool.update_scratchpad(self.agent_state.scratchpad)

        # Find and update the tool in our tools list
        for i, tool in enumerate(self.tools):
            if hasattr(tool, 'name') and tool.name == self.scratchpad_tool.name:
                # Update the underlying tool reference
                if hasattr(tool, '_tool'):
                    # Update the wrapped tool's scratchpad data
                    tool._tool.scratchpad_data = self.agent_state.scratchpad
                    tool._tool.description = self.scratchpad_tool.description
                    # Also update the wrapper's description
                    tool.description = self.scratchpad_tool.description
                elif hasattr(tool, 'scratchpad_data'):
                    # Direct update if it's the tool itself
                    tool.scratchpad_data = self.agent_state.scratchpad
                    tool.description = self.scratchpad_tool.description
                break

        # Regenerate tools description to reflect the updated tool
        self.tools_description = render_text_description_and_args(self.tools)

        # Add a message to inform the agent about available scratchpad keys
        if self.agent_state.scratchpad:
            keys_info = self._get_scratchpad_keys_info()
            if keys_info:
                scratchpad_update_msg = (
                    f"\n💾 SCRATCHPAD UPDATE: New data has been stored in your scratchpad memory.\n"
                    f"{keys_info}\n"
                    f"Use 'Access Scratchpad Memory' tool with the exact key name to retrieve any of this data."
                )
                self._append_message(scratchpad_update_msg, role="system")

    def _get_scratchpad_keys_info(self) -> str:
        """Get formatted information about available scratchpad keys."""
        if not self.agent_state.scratchpad:
            return ""

        keys_info = []
        for key, value in self.agent_state.scratchpad.items():
            # Create a brief description of what's stored
            if isinstance(value, dict):
                if 'data' in value and isinstance(value['data'], list):
                    preview = f"list of {len(value['data'])} items"
                else:
                    preview = f"dict with {len(value)} fields"
            elif isinstance(value, list):
                preview = f"list of {len(value)} items"
            elif isinstance(value, str):
                preview = f"string ({len(value)} chars)"
            else:
                preview = type(value).__name__

            keys_info.append(f"  • '{key}': {preview}")

        return "Available keys:\n" + "\n".join(keys_info)

    def _validate_acceptance_criteria(self, output: str) -> Tuple[bool, List[str]]:
        """Validate if the output meets acceptance criteria using enhanced tracking.

        Args:
            output: The final output to validate

        Returns:
            Tuple[bool, List[str]]: (is_valid, list of unmet criteria)
        """
        print(f"[DEBUG] _validate_acceptance_criteria started")
        if not self.agent_state.acceptance_criteria:
            print(f"[DEBUG] No acceptance criteria to validate")
            return True, []

        unmet_criteria = []

        # First, try deterministic validation based on tracked items
        for criterion in self.agent_state.acceptance_criteria:
            progress = self.agent_state.criteria_progress.get(criterion)

            if progress:
                # Use deterministic checks first
                is_met = False
                reason = ""

                # Check if we have expected item counts
                if progress.total_items_expected is not None:
                    if len(progress.processed_items) >= progress.total_items_expected:
                        is_met = True
                    else:
                        reason = f"Only processed {len(progress.processed_items)}/{progress.total_items_expected} items"

                # Check completion percentage
                elif progress.completion_percentage >= 95:  # Allow slight margin
                    is_met = True

                # Check if marked as completed with items processed
                elif progress.status == "completed" and progress.processed_items:
                    is_met = True

                # For criteria without specific tracking, fall back to content analysis
                else:
                    is_met = self._validate_criterion_by_content(criterion, output)

                if not is_met:
                    unmet_criteria.append(criterion)
                    if reason:
                        self._printer.print(
                            content=f"✗ Criterion not met: {criterion} - {reason}",
                            color="yellow"
                        )
                    else:
                        self._printer.print(
                            content=f"✗ Criterion not met: {criterion}",
                            color="yellow"
                        )
                else:
                    self._printer.print(
                        content=f"✓ Criterion met: {criterion}",
                        color="green"
                    )
            else:
                # No progress tracked for this criterion
                if not self._validate_criterion_by_content(criterion, output):
                    unmet_criteria.append(criterion)
                    self._printer.print(
                        content=f"✗ Criterion not met: {criterion} - No progress tracked",
                        color="yellow"
                    )

        print(f"[DEBUG] Validation complete: {len(unmet_criteria)} unmet criteria")

        # Log validation result
        self.agent_state.log_progress(
            action="Validation check",
            result=f"{len(self.agent_state.acceptance_criteria) - len(unmet_criteria)}/{len(self.agent_state.acceptance_criteria)} criteria met"
        )

        return len(unmet_criteria) == 0, unmet_criteria

    def _validate_criterion_by_content(self, criterion: str, output: str) -> bool:
        """Validate a single criterion by analyzing the output content."""
        criterion_lower = criterion.lower()
        output_lower = output.lower()

        # Look for key indicators in the output
        if "all" in criterion_lower or "every" in criterion_lower:
            # For "all" criteria, check if output indicates completeness
            completeness_indicators = ["all", "every", "complete", "total", "full"]
            return any(indicator in output_lower for indicator in completeness_indicators)

        # Check if key terms from criterion appear in output
        important_words = [word for word in criterion_lower.split() if len(word) > 3]
        matches = sum(1 for word in important_words if word in output_lower)

        return matches >= len(important_words) * 0.7  # 70% of important words should match

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
        example_usage = ""
        if self.scratchpad_tool and self.agent_state.scratchpad:
            scratchpad_keys = list(self.agent_state.scratchpad.keys())

            # Create detailed summary of each key
            key_details = []
            for key in scratchpad_keys:
                value = self.agent_state.scratchpad[key]
                if isinstance(value, list):
                    key_details.append(f"  • '{key}': contains {len(value)} items (list)")
                elif isinstance(value, dict):
                    if 'data' in value and isinstance(value['data'], list):
                        key_details.append(f"  • '{key}': contains {len(value['data'])} items (nested list)")
                    else:
                        key_details.append(f"  • '{key}': contains data with {len(value)} fields (dict)")
                else:
                    key_details.append(f"  • '{key}': contains stored data")

            scratchpad_info = f"""
📦 YOUR SCRATCHPAD CONTAINS THE FOLLOWING DATA:
{chr(10).join(key_details)}

🔑 TO ACCESS THIS DATA: Use the "Access Scratchpad Memory" tool with the EXACT key name.
"""

            # Provide specific example based on first key
            if scratchpad_keys:
                example_key = scratchpad_keys[0]
                example_usage = f"""
📋 EXAMPLE - How to retrieve scratchpad data:

Thought: I need to access the {example_key} from my scratchpad
Action: Access Scratchpad Memory
Action Input: {{"key": "{example_key}"}}

⚠️ REMEMBER: Use the EXACT key name as shown above!
"""

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
- Overall progress: {self.agent_state.overall_progress}%
- Criteria progress: {sum(1 for p in self.agent_state.criteria_progress.values() if p.status == 'completed')}/{len(self.agent_state.criteria_progress)} completed"""

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

{scratchpad_info}

💡 WHAT YOU'RE MISSING:
{chr(10).join(f"• {hint}" for hint in missing_data_hints) if missing_data_hints else "• Review the criteria and gather all required data"}

{example_usage}

🔧 YOUR NEXT STEPS:
1. STOP trying to provide a Final Answer
2. ACCESS your scratchpad data FIRST if you haven't already
3. Use additional tools if needed to gather missing information
4. Only provide Final Answer when ALL data is complete
{tools_hint}

⚠️ IMPORTANT REMINDERS:
- The task requires COMPLETE data for EVERY item
- You already have data stored - ACCESS IT using "Access Scratchpad Memory"
- Each item needs ALL requested details (dates, subjects, senders, etc.)
- Continue until you have retrieved and processed ALL required data

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

    def _update_progress_tracking(self) -> None:
        """Analyze current state and update progress tracking for each criterion."""
        if not self.agent_state.acceptance_criteria:
            return

        try:
            # For each criterion, analyze progress deterministically
            for criterion in self.agent_state.acceptance_criteria:
                # Get detailed analysis from scratchpad
                analysis = self.agent_state.analyze_scratchpad_for_criterion_progress(criterion)

                # Get current progress or initialize
                current_progress = self.agent_state.criteria_progress.get(criterion)
                if not current_progress:
                    self.agent_state.initialize_criteria_progress()
                    current_progress = self.agent_state.criteria_progress.get(criterion)

                # Determine status based on analysis
                if analysis["data_completeness"] >= 95:
                    status = "completed"
                elif analysis["data_completeness"] > 0:
                    status = "in_progress"
                else:
                    status = "not_started"

                # Generate specific next steps
                next_steps = self.agent_state.generate_specific_next_steps(criterion)
                remaining_work = " | ".join(next_steps[:2]) if next_steps else "Continue gathering data"

                # Build progress notes
                notes_parts = []
                if analysis["item_count"] > 0:
                    notes_parts.append(f"{analysis['item_count']} items found")
                if analysis["relevant_data"]:
                    notes_parts.append(f"Data in: {', '.join(analysis['relevant_data'][:2])}")
                if analysis["specific_gaps"]:
                    notes_parts.append(f"Gap: {analysis['specific_gaps'][0]}")

                progress_notes = " - ".join(notes_parts) if notes_parts else "No data gathered yet"

                # Extract processed items from analysis
                processed_items = list(analysis["processed_ids"]) if analysis["processed_ids"] else None

                # Update the criterion progress
                self.agent_state.update_criterion_progress(
                    criterion=criterion,
                    status=status,
                    progress_notes=progress_notes,
                    completion_percentage=analysis["data_completeness"],
                    remaining_work=remaining_work,
                    processed_items=processed_items
                )

                # Log the update
                self._printer.print(
                    content=f"Progress Update - {criterion}: {analysis['data_completeness']}% complete",
                    color="cyan"
                )

            # Update focus and next steps based on current progress
            self._update_focus_and_next_steps()

        except Exception as e:
            self._printer.print(
                content=f"Error updating progress tracking: {str(e)}",
                color="yellow"
            )
            # Use fallback analysis
            self._fallback_progress_analysis()
            self._update_focus_and_next_steps()

    def _create_forced_final_answer(self) -> str:
        """Create a forced final answer based on current progress when max iterations are reached."""

        # Start with a note about incomplete execution
        output_parts = ["Note: Task execution was stopped after reaching maximum iterations."]

        # Add progress summary if acceptance criteria exist
        if self.agent_state.acceptance_criteria and self.agent_state.criteria_progress:
            output_parts.append("\n## Progress Summary:")
            for criterion, progress in self.agent_state.criteria_progress.items():
                status_icon = "✅" if progress.status == "completed" else "🔄" if progress.status == "in_progress" else "❌"
                output_parts.append(f"{status_icon} {criterion} ({progress.completion_percentage}%)")
                if progress.progress_notes:
                    output_parts.append(f"   - {progress.progress_notes}")

        # Add available data from scratchpad
        if self.agent_state.scratchpad:
            output_parts.append("\n## Available Data:")
            for key, value in self.agent_state.scratchpad.items():
                if isinstance(value, list):
                    output_parts.append(f"\n### {key} ({len(value)} items):")
                    # Show first few items
                    for i, item in enumerate(value[:5]):
                        output_parts.append(f"{i+1}. {self._format_item_for_output(item)}")
                    if len(value) > 5:
                        output_parts.append(f"... and {len(value) - 5} more items")
                elif isinstance(value, dict):
                    output_parts.append(f"\n### {key}:")
                    output_parts.append(self._format_item_for_output(value))
                else:
                    output_parts.append(f"\n### {key}:")
                    output_parts.append(str(value)[:200] + "..." if len(str(value)) > 200 else str(value))

        # If no data available, check recent messages for any useful information
        if not self.agent_state.scratchpad and self.messages:
            output_parts.append("\n## Recent Activity:")
            # Look for the last few assistant messages that might contain results
            assistant_messages = [msg for msg in self.messages[-10:] if msg.get("role") == "assistant"]
            for msg in assistant_messages[-3:]:
                content = msg.get("content", "")
                if "Observation:" in content:
                    # Extract observation content
                    obs_match = re.search(r"Observation:\s*(.+?)(?:\n|$)", content, re.DOTALL)
                    if obs_match:
                        output_parts.append(f"- {obs_match.group(1)[:200]}")

        return "\n".join(output_parts)

    def _format_item_for_output(self, item: Any) -> str:
        """Format an item for inclusion in the final output."""
        if isinstance(item, dict):
            # Format dictionary items nicely
            parts = []
            for key in ["subject", "sender", "date", "snippet", "id"]:  # Common email fields
                if key in item:
                    parts.append(f"{key}: {item[key]}")
            if parts:
                return " | ".join(parts)
            else:
                # Generic dict formatting
                return " | ".join(f"{k}: {v}" for k, v in list(item.items())[:3])
        elif isinstance(item, str):
            return item[:100] + "..." if len(item) > 100 else item
        else:
            return str(item)[:100]

    def _update_focus_and_next_steps(self) -> None:
        """Update the agent's focus and next steps based on current progress."""
        incomplete_criteria = [
            (c, p) for c, p in self.agent_state.criteria_progress.items()
            if p.status != "completed"
        ]

        if not incomplete_criteria:
            # All criteria completed
            self.agent_state.set_focus_and_next_steps(
                focus="All acceptance criteria met - ready to provide final answer",
                next_steps=["Compile all gathered data into the required format", "Provide comprehensive final answer"]
            )
        else:
            # Focus on the least progressed criterion
            least_progress = min(incomplete_criteria, key=lambda x: x[1].completion_percentage)
            criterion_name, progress = least_progress

            if progress.completion_percentage == 0:
                focus = f"Need to start working on: {criterion_name}"
                next_steps = [
                    f"Use tools to gather data for: {criterion_name}",
                    "Store results in scratchpad for later access"
                ]
            else:
                focus = f"Continue working on: {criterion_name} (currently {progress.completion_percentage}% complete)"
                next_steps = [
                    f"Complete remaining work: {progress.remaining_work}",
                    "Verify data completeness before moving to next criterion"
                ]

                # Add specific guidance based on scratchpad content
                if self.agent_state.scratchpad:
                    next_steps.append("Access scratchpad data to build on existing progress")

            self.agent_state.set_focus_and_next_steps(focus, next_steps)

    def _fallback_progress_analysis(self) -> None:
        """Fallback progress analysis using simple keyword matching."""
        for criterion in self.agent_state.acceptance_criteria:
            # Use scratchpad analysis to determine progress
            analysis = self.agent_state.analyze_scratchpad_for_criterion_progress(criterion)

            # Determine status and progress
            if analysis["data_completeness"] >= 90:
                status = "completed"
                progress = 100
                remaining = ""
            elif analysis["data_completeness"] > 0:
                status = "in_progress"
                progress = analysis["data_completeness"]
                remaining = self._determine_remaining_work(criterion, analysis)
            else:
                status = "not_started"
                progress = 0
                remaining = "Need to gather data for this criterion"

            # Build progress notes
            notes = ""
            if analysis["relevant_data"]:
                notes = f"Found data in: {', '.join(analysis['relevant_data'])}"

            # Update the criterion progress
            self.agent_state.update_criterion_progress(
                criterion=criterion,
                status=status,
                progress_notes=notes,
                completion_percentage=progress,
                remaining_work=remaining
            )

    def _determine_remaining_work(self, criterion: str, analysis: Dict[str, Any]) -> str:
        """Determine what work remains for a criterion based on analysis."""
        criterion_lower = criterion.lower()

        # Analyze based on common patterns
        if "all" in criterion_lower or "every" in criterion_lower:
            if analysis["relevant_data"]:
                return "Ensure all items are included, not just a subset"
            return "Need to gather comprehensive data covering all items"

        if "date" in criterion_lower or "time" in criterion_lower:
            return "Include complete timestamp information"

        if "format" in criterion_lower:
            return "Format data according to requirements"

        return "Complete data gathering for this criterion"
