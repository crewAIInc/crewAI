from typing import Any, List, Optional

from agents import Agent as OpenAIAgent
from agents import Runner, Tool, enable_verbose_stdout_logging
from pydantic import Field, PrivateAttr

from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai.agents.agent_adapters.openai_agents.structured_output_adapter import (
    OpenAIConverterAdapter,
)
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import BaseTool
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities import Logger
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.agent_events import (
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)

from .openai_agent_tool_adapter import OpenAIAgentToolAdapter


class OpenAIAgentAdapter(BaseAgentAdapter):
    """Adapter for OpenAI Assistants"""

    model_config = {"arbitrary_types_allowed": True}

    _openai_agent: OpenAIAgent = PrivateAttr()
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger())
    _active_thread: Optional[str] = PrivateAttr(default=None)
    function_calling_llm: Any = Field(default=None)
    step_callback: Any = Field(default=None)
    converted_tools: Optional[List[Tool]] = Field(default=None)
    _tool_adapter: OpenAIAgentToolAdapter = PrivateAttr()
    _converter_adapter: OpenAIConverterAdapter = PrivateAttr()

    def __init__(
        self,
        openai_agent: OpenAIAgent,
        model: str = "gpt-4o-mini",
        tools: Optional[List[BaseTool]] = None,
        **kwargs,
    ):
        super().__init__(
            role=openai_agent.name,
            goal=openai_agent.instructions,
            backstory=openai_agent.instructions,
            **kwargs,
        )
        self._openai_agent = openai_agent
        self._openai_agent.model = model
        self.tools = tools
        self._tool_adapter = OpenAIAgentToolAdapter(tools=tools)
        self.llm = model
        self._converter_adapter = OpenAIConverterAdapter(self)

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """Execute a task using the OpenAI Assistant"""
        self.create_agent_executor(tools)
        self._converter_adapter.configure_structured_output(task)

        if self.verbose:
            enable_verbose_stdout_logging()

        try:
            task_prompt = task.prompt()
            if context:
                task_prompt = self.i18n.slice("task_with_context").format(
                    task=task_prompt, context=context
                )
            crewai_event_bus.emit(
                self,
                event=AgentExecutionStartedEvent(
                    agent=self,
                    tools=self.tools,
                    task_prompt=task_prompt,
                    task=task,
                ),
            )
            # This is pretty much the agent_executor logic:
            result = self.agent_executor.run_sync(self._openai_agent, task_prompt)
            return self.handle_execution_result(result)

        except Exception as e:
            self._logger.log("error", f"Error executing OpenAI task: {str(e)}")
            crewai_event_bus.emit(
                self,
                event=AgentExecutionErrorEvent(
                    agent=self,
                    task=task,
                    error=str(e),
                ),
            )
            raise

    def create_agent_executor(self, tools: Optional[List[BaseTool]] = None) -> None:
        """
        Configure the OpenAI agent for execution.
        While OpenAI handles execution differently through Runner,
        we can use this method to set up tools and configurations.
        """
        all_tools = list(self.tools or []) + list(tools or [])

        if all_tools:
            self.configure_tools(all_tools)

        self.agent_executor = Runner

    def configure_tools(self, tools: Optional[List[BaseTool]] = None) -> None:
        """Configure tools for the OpenAI Assistant"""
        if tools:
            self._tool_adapter.configure_tools(tools)
            if self._tool_adapter.converted_tools:
                self._openai_agent.tools = self._tool_adapter.converted_tools

    def handle_execution_result(self, result: Any) -> str:
        """Process OpenAI Assistant execution result converting any structured output to a string"""
        return self._converter_adapter.post_process_result(result.final_output)

    def get_delegation_tools(self, agents: List[BaseAgent]) -> List[BaseTool]:
        """Implement delegation tools support"""
        agent_tools = AgentTools(agents=agents)
        tools = agent_tools.tools()
        return tools

    def get_output_converter(
        self, llm: Any, text: str, model: Any, instructions: str
    ) -> Any:
        """Convert output format if needed"""
        from crewai.utilities.converter import Converter

        return Converter(llm=llm, text=text, model=model, instructions=instructions)

    def _parse_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Parse and validate tools"""
        tools_list = []
        try:
            # tentatively try to import from crewai_tools import BaseTool as CrewAITool
            from crewai.tools import BaseTool as CrewAITool

            for tool in tools:
                if isinstance(tool, CrewAITool):
                    tools_list.append(tool.to_structured_tool())
                else:
                    tools_list.append(tool)
        except ModuleNotFoundError:
            tools_list = []
            for tool in tools:
                tools_list.append(tool)

        return tools_list

    def configure_structured_output(self, task) -> None:
        """Configure the structured output for the specific agent implementation.

        Args:
            structured_output: The structured output to be configured
        """
        self._converter_adapter.configure_structured_output(task)
