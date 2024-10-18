import json
import re
from typing import Any, Dict, List, Union

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
from crewai.tools.tool_usage import ToolUsage, ToolUsageErrorException
from crewai.utilities import I18N, Printer
from crewai.utilities.constants import TRAINING_DATA_FILE
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)
from crewai.utilities.logger import Logger
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
        tools: List[Any],
        tools_names: str,
        stop_words: List[str],
        tools_description: str,
        tools_handler: ToolsHandler,
        step_callback: Any = None,
        original_tools: List[Any] = [],
        function_calling_llm: Any = None,
        respect_context_window: bool = False,
        request_within_rpm_limit: Any = None,
        callbacks: List[Any] = [],
    ):
        self._i18n: I18N = I18N()
        self.llm = llm
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
        self.have_forced_answer = False
        self.name_to_tool_map = {tool.name: tool for tool in self.tools}
        if self.llm.stop:
            self.llm.stop = list(set(self.llm.stop + self.stop))
        else:
            self.llm.stop = self.stop

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
        formatted_answer = self._invoke_loop()

        if self.ask_for_human_input:
            human_feedback = self._ask_human_input(formatted_answer.output)
            if self.crew and self.crew._train:
                self._handle_crew_training_output(formatted_answer, human_feedback)

            # Making sure we only ask for it once, so disabling for the next thought loop
            self.ask_for_human_input = False
            self.messages.append(self._format_msg(f"Feedback: {human_feedback}"))
            formatted_answer = self._invoke_loop()

            if self.crew and self.crew._train:
                self._handle_crew_training_output(formatted_answer)
        self._create_short_term_memory(formatted_answer)
        self._create_long_term_memory(formatted_answer)
        return {"output": formatted_answer.output}

    def _invoke_loop(self, formatted_answer=None):
        try:
            while not isinstance(formatted_answer, AgentFinish):
                if not self.request_within_rpm_limit or self.request_within_rpm_limit():
                    answer = self.llm.call(
                        self.messages,
                        callbacks=self.callbacks,
                    )

                    if not self.use_stop_words:
                        try:
                            self._format_answer(answer)
                        except OutputParserException as e:
                            if (
                                FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE
                                in e.error
                            ):
                                answer = answer.split("Observation:")[0].strip()

                    self.iterations += 1
                    formatted_answer = self._format_answer(answer)

                    if isinstance(formatted_answer, AgentAction):
                        action_result = self._use_tool(formatted_answer)
                        formatted_answer.text += f"\nObservation: {action_result}"
                        formatted_answer.result = action_result
                        self._show_logs(formatted_answer)

                        if self.step_callback:
                            self.step_callback(formatted_answer)

                        if self._should_force_answer():
                            if self.have_forced_answer:
                                return AgentFinish(
                                    output=self._i18n.errors(
                                        "force_final_answer_error"
                                    ).format(formatted_answer.text),
                                    text=formatted_answer.text,
                                )
                            else:
                                formatted_answer.text += (
                                    f'\n{self._i18n.errors("force_final_answer")}'
                                )
                                self.have_forced_answer = True
                        self.messages.append(
                            self._format_msg(formatted_answer.text, role="assistant")
                        )

        except OutputParserException as e:
            self.messages.append({"role": "user", "content": e.error})
            if self.iterations > self.log_error_after:
                self._printer.print(
                    content=f"Error parsing LLM output, agent will retry: {e.error}",
                    color="red",
                )
            return self._invoke_loop(formatted_answer)

        except Exception as e:
            if LLMContextLengthExceededException(str(e))._is_context_limit_error(
                str(e)
            ):
                self._handle_context_length()
                return self._invoke_loop(formatted_answer)
            else:
                raise e

        self._show_logs(formatted_answer)
        return formatted_answer

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
            self._printer.print(
                content=f"\033[95m## Task:\033[00m \033[92m{self.task.description}\033[00m"
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

    def _use_tool(self, agent_action: AgentAction) -> Any:
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
        tool_calling = tool_usage.parse(agent_action.text)

        if isinstance(tool_calling, ToolUsageErrorException):
            tool_result = tool_calling.message
        else:
            if tool_calling.tool_name.casefold().strip() in [
                name.casefold().strip() for name in self.name_to_tool_map
            ] or tool_calling.tool_name.casefold().replace("_", " ") in [
                name.casefold().strip() for name in self.name_to_tool_map
            ]:
                tool_result = tool_usage.use(tool_calling, agent_action.text)
            else:
                tool_result = self._i18n.errors("wrong_tool_name").format(
                    tool=tool_calling.tool_name,
                    tools=", ".join([tool.name.casefold() for tool in self.tools]),
                )
        return tool_result

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
                        self._i18n.slice("sumamrize_instruction").format(group=group),
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
            self._logger.log(
                "debug",
                "Context length exceeded. Summarizing content to fit the model context window.",
                color="yellow",
            )
            self._summarize_messages()
        else:
            self._logger.log(
                "debug",
                "Context length exceeded. Consider using smaller text or RAG tools from crewai_tools.",
                color="red",
            )
            raise SystemExit(
                "Context length exceeded and user opted not to summarize. Consider using smaller text or RAG tools from crewai_tools."
            )

    def _handle_crew_training_output(
        self, result: AgentFinish, human_feedback: str | None = None
    ) -> None:
        """Function to handle the process of the training data."""
        agent_id = str(self.agent.id)  # type: ignore

        # Load training data
        training_handler = CrewTrainingHandler(TRAINING_DATA_FILE)
        training_data = training_handler.load()

        # Check if training data exists, human input is not requested, and self.crew is valid
        if training_data and not self.ask_for_human_input:
            if self.crew is not None and hasattr(self.crew, "_train_iteration"):
                train_iteration = self.crew._train_iteration
                if agent_id in training_data and isinstance(train_iteration, int):
                    training_data[agent_id][train_iteration][
                        "improved_output"
                    ] = result.output
                    training_handler.save(training_data)
                else:
                    self._logger.log(
                        "error",
                        "Invalid train iteration type or agent_id not in training data.",
                        color="red",
                    )
            else:
                self._logger.log(
                    "error",
                    "Crew is None or does not have _train_iteration attribute.",
                    color="red",
                )

        if self.ask_for_human_input and human_feedback is not None:
            training_data = {
                "initial_output": result.output,
                "human_feedback": human_feedback,
                "agent": agent_id,
                "agent_role": self.agent.role,  # type: ignore
            }
            if self.crew is not None and hasattr(self.crew, "_train_iteration"):
                train_iteration = self.crew._train_iteration
                if isinstance(train_iteration, int):
                    CrewTrainingHandler(TRAINING_DATA_FILE).append(
                        train_iteration, agent_id, training_data
                    )
                else:
                    self._logger.log(
                        "error",
                        "Invalid train iteration type. Expected int.",
                        color="red",
                    )
            else:
                self._logger.log(
                    "error",
                    "Crew is None or does not have _train_iteration attribute.",
                    color="red",
                )

    def _format_prompt(self, prompt: str, inputs: Dict[str, str]) -> str:
        prompt = prompt.replace("{input}", inputs["input"])
        prompt = prompt.replace("{tool_names}", inputs["tool_names"])
        prompt = prompt.replace("{tools}", inputs["tools"])
        return prompt

    def _format_answer(self, answer: str) -> Union[AgentAction, AgentFinish]:
        return CrewAgentParser(agent=self.agent).parse(answer)

    def _format_msg(self, prompt: str, role: str = "user") -> Dict[str, str]:
        return {"role": role, "content": prompt}
