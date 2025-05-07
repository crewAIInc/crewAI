from typing import Any, List, Optional

from pydantic import Field, PrivateAttr

from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai.agents.agent_adapters.openai_agents.structured_output_converter import (
    OpenAIConverterAdapter,
)
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools import BaseTool
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities import Logger
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)

try:
    from agents import Agent as OpenAIAgent  # type: ignore
    from agents import Runner, enable_verbose_stdout_logging  # type: ignore

    from .openai_agent_tool_adapter import OpenAIAgentToolAdapter

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIAgentAdapter(BaseAgentAdapter):
    """Adapter for OpenAI Assistants"""

    model_config = {"arbitrary_types_allowed": True}

    _openai_agent: "OpenAIAgent" = PrivateAttr()
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger())
    _active_thread: Optional[str] = PrivateAttr(default=None)
    function_calling_llm: Any = Field(default=None)
    step_callback: Any = Field(default=None)
    _tool_adapter: "OpenAIAgentToolAdapter" = PrivateAttr()
    _converter_adapter: OpenAIConverterAdapter = PrivateAttr()

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        tools: Optional[List[BaseTool]] = None,
        agent_config: Optional[dict] = None,
        **kwargs,
    ):
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI Agent Dependencies are not installed. Please install it using `uv add openai-agents`"
            )
        else:
            role = kwargs.pop("role", None)
            goal = kwargs.pop("goal", None)
            backstory = kwargs.pop("backstory", None)
            super().__init__(
                role=role,
                goal=goal,
                backstory=backstory,
                tools=tools,
                agent_config=agent_config,
                **kwargs,
            )
            self._tool_adapter = OpenAIAgentToolAdapter(tools=tools)
            self.llm = model
            self._converter_adapter = OpenAIConverterAdapter(self)

    def _build_system_prompt(self) -> str:
        """Build a system prompt for the OpenAI agent."""
        base_prompt = f"""
            You are {self.role}.
        
            Your goal is: {self.goal}

            Your backstory: {self.backstory}

            When working on tasks, think step-by-step and use the available tools when necessary.
        """
        return self._converter_adapter.enhance_system_prompt(base_prompt)

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """Execute a task using the OpenAI Assistant"""
        self._converter_adapter.configure_structured_output(task)
        self.create_agent_executor(tools)

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
            result = self.agent_executor.run_sync(self._openai_agent, task_prompt)
            final_answer = self.handle_execution_result(result)
            crewai_event_bus.emit(
                self,
                event=AgentExecutionCompletedEvent(
                    agent=self, task=task, output=final_answer
                ),
            )
            return final_answer

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

        instructions = self._build_system_prompt()
        self._openai_agent = OpenAIAgent(
            name=self.role,
            instructions=instructions,
            model=self.llm,
            **self._agent_config or {},
        )

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

    def configure_structured_output(self, task) -> None:
        """Configure the structured output for the specific agent implementation.

        Args:
            structured_output: The structured output to be configured
        """
        self._converter_adapter.configure_structured_output(task)
