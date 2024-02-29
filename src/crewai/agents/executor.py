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
from crewai.tools.tool_usage import ToolUsage, ToolUsageErrorException
from crewai.utilities import I18N


class CrewAgentExecutor(AgentExecutor):
    # Define class attributes
    _i18n: I18N = I18N()  # Internationalization object for multilingual support
    llm: Any = None  # Language model used by the agent
    iterations: int = 0  # Counter for the number of iterations performed by the agent
    task: Any = None  # Task that the agent is currently working on
    tools_description: str = ""  # Description of the tools used by the agent
    tools_names: str = ""  # Names of the tools used by the agent
    function_calling_llm: Any = None  # Function that calls the language model
    request_within_rpm_limit: Any = None  # Function that checks if the request is within the RPM limit
    tools_handler: InstanceOf[ToolsHandler] = None  # Handler for the tools used by the agent
    max_iterations: Optional[int] = 15  # Maximum number of iterations allowed for the agent
    force_answer_max_iterations: Optional[int] = None  # Maximum number of iterations after which the agent is forced to answer
    step_callback: Optional[Any] = None  # Callback function to be called after each step

    @root_validator()
    def set_force_answer_max_iterations(cls, values: Dict) -> Dict:
        # Set the maximum number of iterations after which the agent is forced to answer
        # This is set to be two less than the maximum number of iterations allowed
        values["force_answer_max_iterations"] = values["max_iterations"] - 2
        return values

    def _should_force_answer(self) -> bool:
        # Check if the agent should be forced to answer
        # This is true if the number of iterations performed by the agent is equal to the maximum number of iterations after which the agent is forced to answer
        return True if self.iterations == self.force_answer_max_iterations else False

    def _call(
        self,
        inputs: Dict[str, str],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Run the given inputs through the agent and get the agent's response.
        This involves running the agent loop until it returns something.
        """

        # Construct a mapping of tool name to tool for easy lookup
        name_to_tool_map = {tool.name: tool for tool in self.tools}

        # We construct a mapping from each tool to a color, used for logging.
        color_mapping = get_color_mapping(
            [tool.name for tool in self.tools], excluded_colors=["green", "red"]
        )

        # List to store the intermediate steps taken by the agent
        intermediate_steps: List[Tuple[AgentAction, str]] = []

        # Initialize the number of iterations and time elapsed
        self.iterations = 0
        time_elapsed = 0.0
        start_time = time.time()

        # Enter the agent loop and continue until the agent returns something
        while self._should_continue(self.iterations, time_elapsed):
            if not self.request_within_rpm_limit or self.request_within_rpm_limit():
                # Take the next step and get the output
                next_step_output = self._take_next_step(
                    name_to_tool_map,
                    color_mapping,
                    inputs,
                    intermediate_steps,
                    run_manager=run_manager,
                )

                # If a step callback is defined, call it with the next step output
                if self.step_callback:
                    self.step_callback(next_step_output)

                # If the next step output indicates that the agent has finished, return the output
                if isinstance(next_step_output, AgentFinish):
                    return self._return(
                        next_step_output, intermediate_steps, run_manager=run_manager
                    )

                # Add the next step output to the list of intermediate steps
                intermediate_steps.extend(next_step_output)

                # If there is only one step in the next step output, check if the tool should return directly
                if len(next_step_output) == 1:
                    next_step_action = next_step_output[0]
                    tool_return = self._get_tool_return(next_step_action)
                    if tool_return is not None:
                        return self._return(
                            tool_return, intermediate_steps, run_manager=run_manager
                        )

                # Increment the number of iterations and update the time elapsed
                self.iterations += 1
                time_elapsed = time.time() - start_time

        # If the agent loop has exited without returning anything, get the stopped response from the agent and return it
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
            """
            This function represents a single iteration in the agent's thought-action-observation loop.
            It is responsible for making decisions based on the current state and executing actions accordingly.
            This function can be overridden to customize the agent's decision-making and action execution process.

            Parameters:
            name_to_tool_map: A dictionary mapping tool names to tool objects.
            color_mapping: A dictionary mapping tool names to colors for logging purposes.
            inputs: A dictionary of inputs to the agent.
            intermediate_steps: A list of tuples representing the intermediate steps taken by the agent.
            run_manager: An optional CallbackManagerForChainRun object for managing callbacks during the run.

            Returns:
            An iterator over the agent's actions and observations during this step.
            """

            try:
                # Prepare the intermediate steps for processing
                intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

                # Call the language model to decide what action to take next
                output = self.agent.plan(
                    intermediate_steps,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **inputs,
                )

                # If the agent should be forced to provide an answer, handle the output accordingly
                if self._should_force_answer():
                    if isinstance(output, AgentFinish):
                        yield output
                        return

                    if isinstance(output, AgentAction):
                        output = output
                    else:
                        raise ValueError(
                            f"Unexpected output type from agent: {type(output)}"
                        )

                    yield AgentStep(
                        action=output, observation=self._i18n.errors("force_final_answer")
                    )
                    return

            except OutputParserException as e:
                # Handle output parsing errors
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
                text = str(e)
                if isinstance(self.handle_parsing_errors, bool):
                    if e.send_to_llm:
                        observation = f"\n{str(e.observation)}"
                        text = str(e.llm_output)
                    else:
                        observation = ""
                elif isinstance(self.handle_parsing_errors, str):
                    observation = f"\n{self.handle_parsing_errors}"
                elif callable(self.handle_parsing_errors):
                    observation = f"\n{self.handle_parsing_errors(e)}"
                else:
                    raise ValueError("Got unexpected type of `handle_parsing_errors`")
                output = AgentAction("_Exception", observation, text)
                if run_manager:
                    run_manager.on_agent_action(output, color="green")
                tool_run_kwargs = self.agent.tool_run_logging_kwargs()
                observation = ExceptionTool().run(
                    output.tool_input,
                    verbose=self.verbose,
                    color=None,
                    callbacks=run_manager.get_child() if run_manager else None,
                    **tool_run_kwargs,
                )

                if self._should_force_answer():
                    yield AgentStep(
                        action=output, observation=self._i18n.errors("force_final_answer")
                    )
                    return

                yield AgentStep(action=output, observation=observation)
                return

            # If the chosen tool is the finishing tool, end the loop and return
            if isinstance(output, AgentFinish):
                yield output
                return

            actions: List[AgentAction]
            actions = [output] if isinstance(output, AgentAction) else output
            yield from actions
            for agent_action in actions:
                if run_manager:
                    run_manager.on_agent_action(agent_action, color="green")
                # If the chosen tool is not the finishing tool, look up the tool and use it
                tool_usage = ToolUsage(
                    tools_handler=self.tools_handler,
                    tools=self.tools,
                    tools_description=self.tools_description,
                    tools_names=self.tools_names,
                    function_calling_llm=self.function_calling_llm,
                    llm=self.llm,
                    task=self.task,
                )
                tool_calling = tool_usage.parse(agent_action.log)

                if isinstance(tool_calling, ToolUsageErrorException):
                    observation = tool_calling.message
                else:
                    if tool_calling.tool_name.lower().strip() in [
                        name.lower().strip() for name in name_to_tool_map
                    ]:
                        observation = tool_usage.use(tool_calling, agent_action.log)
                    else:
                        observation = self._i18n.errors("wrong_tool_name").format(
                            tool=tool_calling.tool_name,
                            tools=", ".join([tool.name for tool in self.tools]),
                        )
                yield AgentStep(action=agent_action, observation=observation)
