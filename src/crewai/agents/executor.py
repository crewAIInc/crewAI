import threading
import time
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union
import click


from langchain.agents import AgentExecutor
from langchain.agents.agent import ExceptionTool
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.exceptions import OutputParserException
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from pydantic import InstanceOf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from crewai.agents.tools_handler import ToolsHandler


from crewai.tools.tool_usage import ToolUsage, ToolUsageErrorException
from crewai.utilities import I18N
from crewai.utilities.constants import TRAINING_DATA_FILE
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)
from crewai.utilities.training_handler import CrewTrainingHandler
from crewai.utilities.logger import Logger


class CrewAgentExecutor(AgentExecutor, CrewAgentExecutorMixin):
    _i18n: I18N = I18N()
    should_ask_for_human_input: bool = False
    llm: Any = None
    iterations: int = 0
    task: Any = None
    tools_description: str = ""
    tools_names: str = ""
    original_tools: List[Any] = []
    crew_agent: Any = None
    crew: Any = None
    function_calling_llm: Any = None
    request_within_rpm_limit: Any = None
    tools_handler: Optional[InstanceOf[ToolsHandler]] = None
    max_iterations: Optional[int] = 15
    have_forced_answer: bool = False
    force_answer_max_iterations: Optional[int] = None  # type: ignore # Incompatible types in assignment (expression has type "int | None", base class "CrewAgentExecutorMixin" defined the type as "int")
    step_callback: Optional[Any] = None
    system_template: Optional[str] = None
    prompt_template: Optional[str] = None
    response_template: Optional[str] = None
    _logger: Logger = Logger()
    _fit_context_window_strategy: Optional[Literal["summarize"]] = "summarize"

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run text through and get agent response."""
        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}
        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name.casefold() for tool in self.tools],
            excluded_colors=["green", "red"],
        )
        intermediate_steps: List[Tuple[AgentAction, str]] = []
        # Allowing human input given task setting
        if self.task and self.task.human_input:
            self.should_ask_for_human_input = True

        # Let's start tracking the number of iterations and time elapsed
        self.iterations = 0
        time_elapsed = 0.0
        start_time = time.time()

        # We now enter the agent loop (until it returns something).
        while self._should_continue(self.iterations, time_elapsed):
            if not self.request_within_rpm_limit or self.request_within_rpm_limit():
                next_step_output = self._take_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager=run_manager,
                )

                if self.step_callback:
                    self.step_callback(next_step_output)

                if isinstance(next_step_output, AgentFinish):
                    # Creating long term memory
                    create_long_term_memory = threading.Thread(
                        target=self._create_long_term_memory, args=(next_step_output,)
                    )
                    create_long_term_memory.start()

                    return self._return(
                        next_step_output, intermediate_steps, run_manager=run_manager
                    )

                intermediate_steps.extend(next_step_output)

                if len(next_step_output) == 1:
                    next_step_action = next_step_output[0]
                    # See if tool should return directly
                    tool_return = self._get_tool_return(next_step_action)
                    if tool_return is not None:
                        return self._return(
                            tool_return, intermediate_steps, run_manager=run_manager
                        )

                self.iterations += 1
                time_elapsed = time.time() - start_time
        output = self.agent.return_stopped_response(
            self.early_stopping_method, intermediate_steps, **inputs
        )

        return self._return(output, intermediate_steps, run_manager=run_manager)

    def _iter_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Iterator[Union[AgentFinish, AgentAction, AgentStep]]:
        """Take a single step in the thought-action-observation loop.

        Override this to take control of how the agent makes and acts on choices.
        """
        try:
            if self._should_force_answer():
                error = self._i18n.errors("force_final_answer")
                output = AgentAction("_Exception", error, error)
                self.have_forced_answer = True
                yield AgentStep(action=output, observation=error)
                return

            intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

            # Call the LLM to see what to do.
            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )

        except OutputParserException as e:
            if isinstance(self.handle_parsing_errors, bool):
                raise_error = not self.handle_parsing_errors
            else:
                raise_error = False
            if raise_error:
                raise ValueError(
                    "An output parsing error occurred. "
                    "In order to pass this error back to the agent and have it try "
                    "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                    f"This is the error: {str(e)}"
                )
            str(e)
            if isinstance(self.handle_parsing_errors, bool):
                if e.send_to_llm:
                    observation = f"\n{str(e.observation)}"
                    str(e.llm_output)
                else:
                    observation = ""
            elif isinstance(self.handle_parsing_errors, str):
                observation = f"\n{self.handle_parsing_errors}"
            elif callable(self.handle_parsing_errors):
                observation = f"\n{self.handle_parsing_errors(e)}"
            else:
                raise ValueError("Got unexpected type of `handle_parsing_errors`")
            output = AgentAction("_Exception", observation, "")

            if run_manager:
                run_manager.on_agent_action(output, color="green")

            tool_run_kwargs = self.agent.tool_run_logging_kwargs()
            observation = ExceptionTool().run(
                output.tool_input,
                verbose=False,
                color=None,
                callbacks=run_manager.get_child() if run_manager else None,
                **tool_run_kwargs,
            )

            if self._should_force_answer():
                error = self._i18n.errors("force_final_answer")
                output = AgentAction("_Exception", error, error)
                yield AgentStep(action=output, observation=error)
                return

            yield AgentStep(action=output, observation=observation)
            return

        except Exception as e:
            if LLMContextLengthExceededException(str(e))._is_context_limit_error(
                str(e)
            ):
                output = self._handle_context_length_error(
                    intermediate_steps, run_manager, inputs
                )

                if isinstance(output, AgentFinish):
                    yield output
                elif isinstance(output, list):
                    for step in output:
                        yield step
                return

            yield AgentStep(
                action=AgentAction("_Exception", str(e), str(e)),
                observation=str(e),
            )
            return

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            if self.should_ask_for_human_input:
                human_feedback = self._ask_human_input(output.return_values["output"])

                if self.crew and self.crew._train:
                    self._handle_crew_training_output(output, human_feedback)

                # Making sure we only ask for it once, so disabling for the next thought loop
                self.should_ask_for_human_input = False
                action = AgentAction(
                    tool="Human Input", tool_input=human_feedback, log=output.log
                )

                yield AgentStep(
                    action=action,
                    observation=self._i18n.slice("human_feedback").format(
                        human_feedback=human_feedback
                    ),
                )
                return

            else:
                if self.crew and self.crew._train:
                    self._handle_crew_training_output(output)

                yield output
                return

        self._create_short_term_memory(output)

        actions: List[AgentAction]
        actions = [output] if isinstance(output, AgentAction) else output
        yield from actions

        for agent_action in actions:
            if run_manager:
                run_manager.on_agent_action(agent_action, color="green")

            tool_usage = ToolUsage(
                tools_handler=self.tools_handler,  # type: ignore # Argument "tools_handler" to "ToolUsage" has incompatible type "ToolsHandler | None"; expected "ToolsHandler"
                tools=self.tools,  # type: ignore # Argument "tools" to "ToolUsage" has incompatible type "Sequence[BaseTool]"; expected "list[BaseTool]"
                original_tools=self.original_tools,
                tools_description=self.tools_description,
                tools_names=self.tools_names,
                function_calling_llm=self.function_calling_llm,
                task=self.task,
                agent=self.crew_agent,
                action=agent_action,
            )

            tool_calling = tool_usage.parse(agent_action.log)

            if isinstance(tool_calling, ToolUsageErrorException):
                observation = tool_calling.message
            else:
                if tool_calling.tool_name.casefold().strip() in [
                    name.casefold().strip() for name in name_to_tool_map
                ] or tool_calling.tool_name.casefold().replace("_", " ") in [
                    name.casefold().strip() for name in name_to_tool_map
                ]:
                    observation = tool_usage.use(tool_calling, agent_action.log)
                else:
                    observation = self._i18n.errors("wrong_tool_name").format(
                        tool=tool_calling.tool_name,
                        tools=", ".join([tool.name.casefold() for tool in self.tools]),
                    )
            yield AgentStep(action=agent_action, observation=observation)

    def _handle_crew_training_output(
        self, output: AgentFinish, human_feedback: str | None = None
    ) -> None:
        """Function to handle the process of the training data."""
        agent_id = str(self.crew_agent.id)

        if (
            CrewTrainingHandler(TRAINING_DATA_FILE).load()
            and not self.should_ask_for_human_input
        ):
            training_data = CrewTrainingHandler(TRAINING_DATA_FILE).load()
            if training_data.get(agent_id):
                training_data[agent_id][self.crew._train_iteration][
                    "improved_output"
                ] = output.return_values["output"]
                CrewTrainingHandler(TRAINING_DATA_FILE).save(training_data)

        if self.should_ask_for_human_input and human_feedback is not None:
            training_data = {
                "initial_output": output.return_values["output"],
                "human_feedback": human_feedback,
                "agent": agent_id,
                "agent_role": self.crew_agent.role,
            }
            CrewTrainingHandler(TRAINING_DATA_FILE).append(
                self.crew._train_iteration, agent_id, training_data
            )

    def _handle_context_length(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[Tuple[AgentAction, str]]:
        text = intermediate_steps[0][1]
        original_action = intermediate_steps[0][0]

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"],
            chunk_size=8000,
            chunk_overlap=500,
        )

        if self._fit_context_window_strategy == "summarize":
            docs = text_splitter.create_documents([text])
            self._logger.log(
                "debug",
                "Summarizing Content, it is recommended to use a RAG tool",
                color="bold_blue",
            )
            summarize_chain = load_summarize_chain(
                self.llm, chain_type="map_reduce", verbose=True
            )
            summarized_docs = []
            for doc in docs:
                summary = summarize_chain.invoke(
                    {"input_documents": [doc]}, return_only_outputs=True
                )

                summarized_docs.append(summary["output_text"])

            formatted_results = "\n\n".join(summarized_docs)
            summary_step = AgentStep(
                action=AgentAction(
                    tool=original_action.tool,
                    tool_input=original_action.tool_input,
                    log=original_action.log,
                ),
                observation=formatted_results,
            )
            summary_tuple = (summary_step.action, summary_step.observation)
            return [summary_tuple]

        return intermediate_steps

    def _handle_context_length_error(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        run_manager: Optional[CallbackManagerForChainRun],
        inputs: Dict[str, str],
    ) -> Union[AgentFinish, List[AgentStep]]:
        self._logger.log(
            "debug",
            "Context length exceeded. Asking user if they want to use summarize prompt to fit, this will reduce context length.",
            color="yellow",
        )
        user_choice = click.confirm(
            "Context length exceeded. Do you want to summarize the text to fit models context window?"
        )
        if user_choice:
            self._logger.log(
                "debug",
                "Context length exceeded. Using summarize prompt to fit, this will reduce context length.",
                color="bold_blue",
            )
            intermediate_steps = self._handle_context_length(intermediate_steps)

            output = self.agent.plan(
                intermediate_steps,
                callbacks=run_manager.get_child() if run_manager else None,
                **inputs,
            )

            if isinstance(output, AgentFinish):
                return output
            elif isinstance(output, AgentAction):
                return [AgentStep(action=output, observation=None)]
            else:
                return [AgentStep(action=action, observation=None) for action in output]
        else:
            self._logger.log(
                "debug",
                "Context length exceeded. Consider using smaller text or RAG tools from crewai_tools.",
                color="red",
            )
            raise SystemExit(
                "Context length exceeded and user opted not to summarize. Consider using smaller text or RAG tools from crewai_tools."
            )
