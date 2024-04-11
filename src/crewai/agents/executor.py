import threading
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from langchain.agents import AgentExecutor
from langchain.agents.agent import ExceptionTool
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain_core.agents import AgentAction, AgentFinish, AgentStep
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool
from langchain_core.utils.input import get_color_mapping
from pydantic import InstanceOf

from crewai.agents.tools_handler import ToolsHandler
from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.tools.tool_usage import ToolUsage, ToolUsageErrorException
from crewai.utilities import I18N
from crewai.utilities.converter import ConverterError
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator


class CrewAgentExecutor(AgentExecutor):
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
    tools_handler: InstanceOf[ToolsHandler] = None
    max_iterations: Optional[int] = 15
    have_forced_answer: bool = False
    force_answer_max_iterations: Optional[int] = None
    step_callback: Optional[Any] = None

    @root_validator()
    def set_force_answer_max_iterations(cls, values: Dict) -> Dict:
        values["force_answer_max_iterations"] = values["max_iterations"] - 2
        return values

    def _should_force_answer(self) -> bool:
        return (
            self.iterations == self.force_answer_max_iterations
        ) and not self.have_forced_answer

    def _create_short_term_memory(self, output) -> None:
        if (
            self.crew
            and self.crew.memory
            and "Action: Delegate work to co-worker" not in output.log
        ):
            memory = ShortTermMemoryItem(
                data=output.log,
                agent=self.crew_agent.role,
                metadata={
                    "observation": self.task.description,
                },
            )
            self.crew._short_term_memory.save(memory)

    def _create_long_term_memory(self, output) -> None:
        if self.crew and self.crew.memory:
            ltm_agent = TaskEvaluator(self.crew_agent)
            evaluation = ltm_agent.evaluate(self.task, output.log)

            if isinstance(evaluation, ConverterError):
                return

            long_term_memory = LongTermMemoryItem(
                task=self.task.description,
                agent=self.crew_agent.role,
                quality=evaluation.quality,
                datetime=str(time.time()),
                expected_output=self.task.expected_output,
                metadata={
                    "suggestions": evaluation.suggestions,
                    "quality": evaluation.quality,
                },
            )
            self.crew._long_term_memory.save(long_term_memory)

            for entity in evaluation.entities:
                entity_memory = EntityMemoryItem(
                    name=entity.name,
                    type=entity.type,
                    description=entity.description,
                    relationships="\n".join([f"- {r}" for r in entity.relationships]),
                )
                self.crew._entity_memory.save(entity_memory)

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
        if self.task.human_input:
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

        # If the tool chosen is the finishing tool, then we end and return.
        if isinstance(output, AgentFinish):
            if self.should_ask_for_human_input:
                # Making sure we only ask for it once, so disabling for the next thought loop
                self.should_ask_for_human_input = False
                human_feedback = self._ask_human_input(output.return_values["output"])
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
                tools_handler=self.tools_handler,
                tools=self.tools,
                original_tools=self.original_tools,
                tools_description=self.tools_description,
                tools_names=self.tools_names,
                function_calling_llm=self.function_calling_llm,
                task=self.task,
                action=agent_action,
            )
            tool_calling = tool_usage.parse(agent_action.log)

            if isinstance(tool_calling, ToolUsageErrorException):
                observation = tool_calling.message
            else:
                if tool_calling.tool_name.casefold().strip() in [
                    name.casefold().strip() for name in name_to_tool_map
                ]:
                    observation = tool_usage.use(tool_calling, agent_action.log)
                else:
                    observation = self._i18n.errors("wrong_tool_name").format(
                        tool=tool_calling.tool_name,
                        tools=", ".join([tool.name.casefold() for tool in self.tools]),
                    )

            yield AgentStep(action=agent_action, observation=observation)

    def _ask_human_input(self, final_answer: dict) -> str:
        """Get human input."""
        return input(
            self._i18n.slice("getting_input").format(final_answer=final_answer)
        )
