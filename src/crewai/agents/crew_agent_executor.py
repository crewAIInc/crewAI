import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from crewai.agents.parser import (
    FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE,
    AgentAction,
    AgentFinish,
    CrewAgentParser,
    OutputParserException,
)
from crewai.agents.tools_handler import ToolsHandler
from crewai.llm import LLM
from crewai.tools.base_tool import BaseTool
from crewai.tools.tool_usage import ToolUsage, ToolUsageErrorException
from crewai.utilities import I18N, Printer
from crewai.utilities.constants import MAX_LLM_RETRY, TRAINING_DATA_FILE
from crewai.utilities.events import (
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)
from crewai.utilities.events.tool_usage_events import ToolUsageStartedEvent
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)
from crewai.utilities.logger import Logger
from crewai.utilities.training_handler import CrewTrainingHandler


@dataclass
class ToolResult:
    result: Any
    result_as_answer: bool


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
        tools: List[BaseTool],
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
        self.llm: LLM = llm
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
        self.tool_name_to_tool_map: Dict[str, BaseTool] = {
            tool.name: tool for tool in self.tools
        }
        self.stop = stop_words
        self.llm.stop = list(set(self.llm.stop + self.stop))

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        if "system" in self.prompt:
            system_prompt = self._format_prompt(self.prompt.get("system", ""), inputs)
            user_prompt = self._format_prompt(self.prompt.get("user", ""), inputs)
            self.messages.append(self._format_msg(system_prompt, role="system"))
            self.messages.append(self._format_msg(user_prompt))
        else:
            user_prompt = self._format_prompt(self.prompt.get("prompt", ""), inputs)
            self.messages.append(self._format_msg(user_prompt))

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
            self._handle_unknown_error(e)
            if e.__class__.__module__.startswith("litellm"):
                # Do not retry on litellm errors
                raise e
            else:
                raise e

        if self.ask_for_human_input:
            formatted_answer = self._handle_human_feedback(formatted_answer)

        self._create_short_term_memory(formatted_answer)
        self._create_long_term_memory(formatted_answer)
        return {"output": formatted_answer.output}

    def _invoke_loop(self) -> AgentFinish:
        """
        Main loop to invoke the agent's thought process until it reaches a conclusion
        or the maximum number of iterations is reached.
        """
        formatted_answer = None
        while not isinstance(formatted_answer, AgentFinish):
            try:
                if self._has_reached_max_iterations():
                    formatted_answer = self._handle_max_iterations_exceeded(
                        formatted_answer
                    )
                    break

                self._enforce_rpm_limit()

                answer = self._get_llm_response()
                formatted_answer = self._process_llm_response(answer)

                if isinstance(formatted_answer, AgentAction):
                    tool_result = self._execute_tool_and_check_finality(
                        formatted_answer
                    )
                    formatted_answer = self._handle_agent_action(
                        formatted_answer, tool_result
                    )

                self._invoke_step_callback(formatted_answer)
                self._append_message(formatted_answer.text, role="assistant")

            except OutputParserException as e:
                formatted_answer = self._handle_output_parser_exception(e)

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    # Do not retry on litellm errors
                    raise e
                if self._is_context_length_exceeded(e):
                    self._handle_context_length()
                    continue
                else:
                    self._handle_unknown_error(e)
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

    def _handle_unknown_error(self, exception: Exception) -> None:
        """Handle unknown errors by informing the user."""
        self._printer.print(
            content="An unknown error occurred. Please check the details below.",
            color="red",
        )
        self._printer.print(
            content=f"Error details: {exception}",
            color="red",
        )

    def _has_reached_max_iterations(self) -> bool:
        """Check if the maximum number of iterations has been reached."""
        return self.iterations >= self.max_iter

    def _enforce_rpm_limit(self) -> None:
        """Enforce the requests per minute (RPM) limit if applicable."""
        if self.request_within_rpm_limit:
            self.request_within_rpm_limit()

    def _get_llm_response(self) -> str:
        """Call the LLM and return the response, handling any invalid responses."""
        try:
            answer = self.llm.call(
                self.messages,
                callbacks=self.callbacks,
            )
        except Exception as e:
            self._printer.print(
                content=f"Error during LLM call: {e}",
                color="red",
            )
            raise e

        if not answer:
            self._printer.print(
                content="Received None or empty response from LLM call.",
                color="red",
            )
            raise ValueError("Invalid response from LLM call - None or empty.")

        return answer

    def _process_llm_response(self, answer: str) -> Union[AgentAction, AgentFinish]:
        """Process the LLM response and format it into an AgentAction or AgentFinish."""
        if not self.use_stop_words:
            try:
                # Preliminary parsing to check for errors.
                self._format_answer(answer)
            except OutputParserException as e:
                if FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE in e.error:
                    answer = answer.split("Observation:")[0].strip()

        return self._format_answer(answer)

    def _handle_agent_action(
        self, formatted_answer: AgentAction, tool_result: ToolResult
    ) -> Union[AgentAction, AgentFinish]:
        """Handle the AgentAction, execute tools, and process the results."""
        add_image_tool = self._i18n.tools("add_image")
        if (
            isinstance(add_image_tool, dict)
            and formatted_answer.tool.casefold().strip()
            == add_image_tool.get("name", "").casefold().strip()
        ):
            self.messages.append(tool_result.result)
            return formatted_answer  # Continue the loop

        if self.step_callback:
            self.step_callback(tool_result)

        formatted_answer.text += f"\nObservation: {tool_result.result}"
        formatted_answer.result = tool_result.result

        if tool_result.result_as_answer:
            return AgentFinish(
                thought="",
                output=tool_result.result,
                text=formatted_answer.text,
            )

        self._show_logs(formatted_answer)
        return formatted_answer

    def _invoke_step_callback(self, formatted_answer) -> None:
        """Invoke the step callback if it exists."""
        if self.step_callback:
            self.step_callback(formatted_answer)

    def _append_message(self, text: str, role: str = "assistant") -> None:
        """Append a message to the message list with the given role."""
        self.messages.append(self._format_msg(text, role=role))

    def _handle_output_parser_exception(self, e: OutputParserException) -> AgentAction:
        """Handle OutputParserException by updating messages and formatted_answer."""
        self.messages.append({"role": "user", "content": e.error})

        formatted_answer = AgentAction(
            text=e.error,
            tool="",
            tool_input="",
            thought="",
        )

        if self.iterations > self.log_error_after:
            self._printer.print(
                content=f"Error parsing LLM output, agent will retry: {e.error}",
                color="red",
            )

        return formatted_answer

    def _is_context_length_exceeded(self, exception: Exception) -> bool:
        """Check if the exception is due to context length exceeding."""
        return LLMContextLengthExceededException(
            str(exception)
        )._is_context_limit_error(str(exception))

    def _show_start_logs(self):
        if self.agent is None:
            raise ValueError("Agent cannot be None")
        if self.agent.verbose or (
            hasattr(self, "crew") and getattr(self.crew, "verbose", False)
        ):
            agent_role = self.agent.role.split("\n")[0]
            self._printer.print(
                content=f"\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
            )
            description = (
                getattr(self.task, "description") if self.task else "Not Found"
            )
            self._printer.print(
                content=f"\033[95m## Task:\033[00m \033[92m{description}\033[00m"
            )

    def _show_logs(self, formatted_answer: Union[AgentAction, AgentFinish]):
        if self.agent is None:
            raise ValueError("Agent cannot be None")
        if self.agent.verbose or (
            hasattr(self, "crew") and getattr(self.crew, "verbose", False)
        ):
            agent_role = self.agent.role.split("\n")[0]
            if isinstance(formatted_answer, AgentAction):
                thought = re.sub(r"\n+", "\n", formatted_answer.thought)
                formatted_json = json.dumps(
                    formatted_answer.tool_input,
                    indent=2,
                    ensure_ascii=False,
                )
                self._printer.print(
                    content=f"\n\n\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
                )
                if thought and thought != "":
                    self._printer.print(
                        content=f"\033[95m## Thought:\033[00m \033[92m{thought}\033[00m"
                    )
                self._printer.print(
                    content=f"\033[95m## Using tool:\033[00m \033[92m{formatted_answer.tool}\033[00m"
                )
                self._printer.print(
                    content=f"\033[95m## Tool Input:\033[00m \033[92m\n{formatted_json}\033[00m"
                )
                self._printer.print(
                    content=f"\033[95m## Tool Output:\033[00m \033[92m\n{formatted_answer.result}\033[00m"
                )
            elif isinstance(formatted_answer, AgentFinish):
                self._printer.print(
                    content=f"\n\n\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
                )
                self._printer.print(
                    content=f"\033[95m## Final Answer:\033[00m \033[92m\n{formatted_answer.output}\033[00m\n\n"
                )

    def _execute_tool_and_check_finality(self, agent_action: AgentAction) -> ToolResult:
        try:
            if self.agent:
                crewai_event_bus.emit(
                    self,
                    event=ToolUsageStartedEvent(
                        agent_key=self.agent.key,
                        agent_role=self.agent.role,
                        tool_name=agent_action.tool,
                        tool_args=agent_action.tool_input,
                        tool_class=agent_action.tool,
                    ),
                )
            tool_usage = ToolUsage(
                tools_handler=self.tools_handler,
                tools=self.tools,
                original_tools=self.original_tools,
                tools_description=self.tools_description,
                tools_names=self.tools_names,
                function_calling_llm=self.function_calling_llm,
                task=self.task,  # type: ignore[arg-type]
                agent=self.agent,
                action=agent_action,
            )
            tool_calling = tool_usage.parse_tool_calling(agent_action.text)

            if isinstance(tool_calling, ToolUsageErrorException):
                tool_result = tool_calling.message
                return ToolResult(result=tool_result, result_as_answer=False)
            else:
                if tool_calling.tool_name.casefold().strip() in [
                    name.casefold().strip() for name in self.tool_name_to_tool_map
                ] or tool_calling.tool_name.casefold().replace("_", " ") in [
                    name.casefold().strip() for name in self.tool_name_to_tool_map
                ]:
                    tool_result = tool_usage.use(tool_calling, agent_action.text)
                    tool = self.tool_name_to_tool_map.get(tool_calling.tool_name)
                    if tool:
                        return ToolResult(
                            result=tool_result, result_as_answer=tool.result_as_answer
                        )
                else:
                    tool_result = self._i18n.errors("wrong_tool_name").format(
                        tool=tool_calling.tool_name,
                        tools=", ".join([tool.name.casefold() for tool in self.tools]),
                    )
                return ToolResult(result=tool_result, result_as_answer=False)

        except Exception as e:
            # TODO: drop
            if self.agent:
                crewai_event_bus.emit(
                    self,
                    event=ToolUsageErrorEvent(  # validation error
                        agent_key=self.agent.key,
                        agent_role=self.agent.role,
                        tool_name=agent_action.tool,
                        tool_args=agent_action.tool_input,
                        tool_class=agent_action.tool,
                        error=str(e),
                    ),
                )
            raise e

    def _summarize_messages(self) -> None:
        messages_groups = []
        for message in self.messages:
            content = message["content"]
            cut_size = self.llm.get_context_window_size()
            for i in range(0, len(content), cut_size):
                messages_groups.append(content[i : i + cut_size])

        summarized_contents = []
        for group in messages_groups:
            summary = self.llm.call(
                [
                    self._format_msg(
                        self._i18n.slice("summarizer_system_message"), role="system"
                    ),
                    self._format_msg(
                        self._i18n.slice("summarize_instruction").format(group=group),
                    ),
                ],
                callbacks=self.callbacks,
            )
            summarized_contents.append(summary)

        merged_summary = " ".join(str(content) for content in summarized_contents)

        self.messages = [
            self._format_msg(
                self._i18n.slice("summary").format(merged_summary=merged_summary)
            )
        ]

    def _handle_context_length(self) -> None:
        if self.respect_context_window:
            self._printer.print(
                content="Context length exceeded. Summarizing content to fit the model context window.",
                color="yellow",
            )
            self._summarize_messages()
        else:
            self._printer.print(
                content="Context length exceeded. Consider using smaller text or RAG tools from crewai_tools.",
                color="red",
            )
            raise SystemExit(
                "Context length exceeded and user opted not to summarize. Consider using smaller text or RAG tools from crewai_tools."
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

    def _format_answer(self, answer: str) -> Union[AgentAction, AgentFinish]:
        return CrewAgentParser(agent=self.agent).parse(answer)

    def _format_msg(self, prompt: str, role: str = "user") -> Dict[str, str]:
        prompt = prompt.rstrip()
        return {"role": role, "content": prompt}

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
            self._format_msg(
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
            self._format_msg(
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

    def _handle_max_iterations_exceeded(self, formatted_answer):
        """
        Handles the case when the maximum number of iterations is exceeded.
        Performs one more LLM call to get the final answer.

        Parameters:
            formatted_answer: The last formatted answer from the agent.

        Returns:
            The final formatted answer after exceeding max iterations.
        """
        self._printer.print(
            content="Maximum iterations reached. Requesting final answer.",
            color="yellow",
        )

        if formatted_answer and hasattr(formatted_answer, "text"):
            assistant_message = (
                formatted_answer.text + f'\n{self._i18n.errors("force_final_answer")}'
            )
        else:
            assistant_message = self._i18n.errors("force_final_answer")

        self.messages.append(self._format_msg(assistant_message, role="assistant"))

        # Perform one more LLM call to get the final answer
        answer = self.llm.call(
            self.messages,
            callbacks=self.callbacks,
        )

        if answer is None or answer == "":
            self._printer.print(
                content="Received None or empty response from LLM call.",
                color="red",
            )
            raise ValueError("Invalid response from LLM call - None or empty.")

        formatted_answer = self._format_answer(answer)
        # Return the formatted answer, regardless of its type
        return formatted_answer
