from typing import Any, List, Optional
import os
import time
import json
from pydantic import Field, PrivateAttr

from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai.agents.agent_adapters.foundry_agents.structured_output_converter import (
    FoundryConverterAdapter,
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
    from azure.ai.projects import AIProjectClient as FoundryClient
    from azure.identity import DefaultAzureCredential
    from azure.ai.agents.models import FunctionTool


    from .foundry_agent_tool_adapter import FoundryAgentToolAdapter

    FOUNDRY_AVAILABLE = True
except ImportError:
    FOUNDRY_AVAILABLE = False


class FoundryAgentAdapter(BaseAgentAdapter):
    """Adapter for Foundry Assistants"""

    model_config = {"arbitrary_types_allowed": True}

    _foundry_client: "FoundryClient" = PrivateAttr()
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger())
    _active_thread_id: Optional[str] = PrivateAttr(default=None)
    function_calling_llm: Any = Field(default=None)
    step_callback: Any = Field(default=None)
    _tool_adapter: "FoundryAgentToolAdapter" = PrivateAttr()
    _converter_adapter: FoundryConverterAdapter = PrivateAttr()
    _converted_tools: Optional[FunctionTool] = PrivateAttr(default=None)
    
    model: str = Field(default="gpt-4.1-mini")

    def __init__(
        self,
        llm: Any = None,
        tools: Optional[List[BaseTool]] = None,
        agent_config: Optional[dict] = None,
        **kwargs,
    ):
        if not FOUNDRY_AVAILABLE:
            raise ImportError(
                "Foundry Agent Dependencies are not installed. Please install it using `uv pip install azure-ai-projects azure-identity`"
            )
        role = kwargs.pop("role", None)
        goal = kwargs.pop("goal", None)
        backstory = kwargs.pop("backstory", None)
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            llm=llm or self.model,
            agent_config=agent_config,
            **kwargs,
        )
        self._tool_adapter = FoundryAgentToolAdapter(tools=tools)
        self._converter_adapter = FoundryConverterAdapter(self)

    def _build_system_prompt(self) -> str:
        """Build a system prompt for the Foundry agent."""
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
        """Execute a task using the Foundry Assistant (automatic mode)"""
        self._converter_adapter.configure_structured_output(task)
        self.create_agent_executor(tools)

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

        try:
            self._foundry_client.agents.create_message(
                thread_id=self._active_thread_id,
                role="user",
                content=task_prompt,
            )

            self._foundry_client.agents.create_and_process_run(
                thread_id=self._active_thread_id,
                agent_id=self._foundry_agent_id
            )

            messages = self._foundry_client.agents.list_messages(thread_id=self._active_thread_id)
            final_answer = messages.data[0].content[0]['text']['value']

            crewai_event_bus.emit(
                self,
                event=AgentExecutionCompletedEvent(
                    agent=self, task=task, output=final_answer
                ),
            )
            return final_answer

        except Exception as e:
            self._logger.log("error", f"Error executing Foundry task: {str(e)}")
            crewai_event_bus.emit(
                self,
                event=AgentExecutionErrorEvent(agent=self, task=task, error=str(e)),
            )
            raise


    def create_agent_executor(self, tools: Optional[List[BaseTool]] = None) -> None:
        self.configure_tools(tools)
        self._foundry_client = FoundryClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        )
        agent = self._foundry_client.agents.create_agent(
            model=self.llm,
            name=self.role or "crewai-agent",
            instructions=self._build_system_prompt(),
            tools=self._converted_tools.definitions if self._converted_tools else None,
        )

        self._foundry_agent_id = agent.id
        self._active_thread_id = self._foundry_client.agents.create_thread().id


    def configure_tools(self, tools: Optional[List[BaseTool]] = None) -> None:
        """Configure tools for the Foundry Assistant"""
        if tools:
            self._tool_adapter.configure_tools(tools)
            if self._tool_adapter.converted_tools:
                self._converted_tools = self._tool_adapter.converted_tools

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
