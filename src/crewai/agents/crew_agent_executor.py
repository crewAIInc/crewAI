import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
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
        existing_stop = self.llm.stop or []
        self.llm.stop = list(
            set(
                existing_stop + self.stop
                if isinstance(existing_stop, list)
                else self.stop
            )
        )

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
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

    def _summarize_messages(self) -> None:
        messages_groups = []
        for message in self.messages:
            content = message["content"]
            cut_size = self.llm.get_context_window_size()
            for i in range(0, len(content), cut_size):
                messages_groups.append({"content": content[i : i + cut_size]})

        summarized_contents = []
        for group in messages_groups:
            summary = self.llm.call(
                [
                    format_message_for_llm(
                        self._i18n.slice("summarizer_system_message"), role="system"
                    ),
                    format_message_for_llm(
                        self._i18n.slice("summarize_instruction").format(
                            group=group["content"]
                        ),
                    ),
                ],
                callbacks=self.callbacks,
            )
            summarized_contents.append({"content": str(summary)})

        merged_summary = " ".join(content["content"] for content in summarized_contents)

        self.messages = [
            format_message_for_llm(
                self._i18n.slice("summary").format(merged_summary=merged_summary)
            )
        ]

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
