from typing import Any, Dict, List

from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from crewai.agents.parser import CrewAgentParser
from crewai.agents.tools_handler import ToolsHandler
from crewai.tools.tool_usage import ToolUsage, ToolUsageErrorException
from crewai.utilities import I18N
from crewai.utilities.constants import TRAINING_DATA_FILE
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)
from crewai.utilities.logger import Logger
from crewai.utilities.training_handler import CrewTrainingHandler
from crewai.llm import LLM
from crewai.agents.parser import AgentAction, AgentFinish, OutputParserException


class CrewAgentExecutor(CrewAgentExecutorMixin):
    _logger: Logger = Logger()

    def __init__(
        self,
        llm: Any,
        task: Any,
        crew: Any,
        agent: Any,
        prompt: str,
        max_iter: int,
        tools: List[Any],
        tools_names: str,
        stop_words: List[str],
        tools_description: str,
        tools_handler: ToolsHandler,
        step_callback: Any = None,
        original_tools: List[Any] = [],
        function_calling_llm: Any = None,
        sliding_context_window: bool = False,
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
        self.tools_handler = tools_handler
        self.original_tools = original_tools
        self.step_callback = step_callback
        self.tools_description = tools_description
        self.function_calling_llm = function_calling_llm
        self.sliding_context_window = sliding_context_window
        self.request_within_rpm_limit = request_within_rpm_limit
        self.should_ask_for_human_input = False
        self.messages = []
        self.iterations = 0
        self.have_forced_answer = False
        self.name_to_tool_map = {tool.name: tool for tool in self.tools}

    def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        formatted_answer = None
        formatted_prompt = self._format_prompt(self.prompt, inputs)
        self.should_ask_for_human_input = inputs.get(
            "should_ask_for_human_input", False
        )
        self.messages = self._messages(formatted_prompt)

        formatted_answer = self._invoke_loop(formatted_answer)

        if self.should_ask_for_human_input:
            human_feedback = self._ask_human_input(formatted_answer.output)
            if self.crew and self.crew._train:
                self._handle_crew_training_output(formatted_answer, human_feedback)

            # Making sure we only ask for it once, so disabling for the next thought loop
            self.should_ask_for_human_input = False
            self.messages.append(
                {"role": "user", "content": f"Feedback: {human_feedback}"}
            )
            formatted_answer = self._invoke_loop(None)

        return {"output": formatted_answer.output}

    def _invoke_loop(self, formatted_answer):
        try:
            while not isinstance(formatted_answer, AgentFinish):
                # print('2222222')
                if not self.request_within_rpm_limit or self.request_within_rpm_limit():
                    # print('3333333')
                    answer = LLM(
                        self.llm, stop=self.stop, callbacks=self.callbacks
                    ).call(self.messages)

                    self.iterations += 1
                    print("*** self.iterations", self.iterations)
                    # if self.iterations > 11:
                    #     sadasd
                    formatted_answer = self._format_answer(answer)

                    if isinstance(formatted_answer, AgentAction):
                        # print('4444444')
                        action_result = self._use_tool(formatted_answer)
                        formatted_answer.text += f"\nObservation: {action_result}"
                        # print(formatted_answer)

                        if self.step_callback:
                            formatted_answer.result = action_result
                            self.step_callback(formatted_answer)
                        if self._should_force_answer():
                            if self.have_forced_answer:
                                return {
                                    "output": self._i18n.errors(
                                        "force_final_answer_error"
                                    ).format(formatted_answer.text)
                                }
                            else:
                                formatted_answer.text += (
                                    f'\n{self._i18n.errors("force_final_answer")}'
                                )
                                self.have_forced_answer = True
                    self.messages.append(
                        {"role": "assistant", "content": formatted_answer.text}
                    )

        except OutputParserException as e:
            # print('5555555')
            self.messages.append({"role": "assistant", "content": e.error})
            self._invoke_loop(formatted_answer)

        except Exception as e:
            # print('6666666')
            print("*** e", e)
            if LLMContextLengthExceededException(str(e))._is_context_limit_error(
                str(e)
            ):
                self._handle_context_length()
                self._invoke_loop(formatted_answer)

        # print('7777777')
        return formatted_answer

    def _use_tool(self, agent_action: AgentAction) -> None:
        tool_usage = ToolUsage(
            tools_handler=self.tools_handler,
            tools=self.tools,
            original_tools=self.original_tools,
            tools_description=self.tools_description,
            tools_names=self.tools_names,
            function_calling_llm=self.function_calling_llm,
            task=self.task,
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
        llm = LLM(self.llm)
        grouped_messages = []

        for message in self.messages:
            content = message["content"]
            for i in range(0, len(content), 5000):
                grouped_messages.append(content[i : i + 5000])

        summarized_contents = []
        for group in grouped_messages:
            summary = llm.call(
                [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes text.",
                    },
                    {
                        "role": "user",
                        "content": f"Summarize the following text, make sure to include all the important information: {group}",
                    },
                ]
            )
            summarized_contents.append(summary)

        merged_summary = " ".join(summarized_contents)

        self.messages = [
            {
                "role": "user",
                "content": f"This is a summary of our conversation so far:\n{merged_summary}",
            }
        ]

    def _handle_context_length(self) -> None:
        if self.sliding_context_window:
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
        agent_id = str(self.agent.id)

        if (
            CrewTrainingHandler(TRAINING_DATA_FILE).load()
            and not self.should_ask_for_human_input
        ):
            training_data = CrewTrainingHandler(TRAINING_DATA_FILE).load()
            if training_data.get(agent_id):
                training_data[agent_id][self.crew._train_iteration][
                    "improved_output"
                ] = result.output
                CrewTrainingHandler(TRAINING_DATA_FILE).save(training_data)

        if self.should_ask_for_human_input and human_feedback is not None:
            training_data = {
                "initial_output": result.output,
                "human_feedback": human_feedback,
                "agent": agent_id,
                "agent_role": self.agent.role,
            }
            CrewTrainingHandler(TRAINING_DATA_FILE).append(
                self.crew._train_iteration, agent_id, training_data
            )

    def _format_prompt(self, prompt: str, inputs: Dict[str, str]) -> str:
        prompt = prompt.replace("{input}", inputs["input"])
        prompt = prompt.replace("{tool_names}", inputs["tool_names"])
        prompt = prompt.replace("{tools}", inputs["tools"])
        return prompt

    def _format_answer(self, answer: str) -> str:
        return CrewAgentParser(agent=self).parse(answer)

    def _messages(self, prompt: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": prompt}]
