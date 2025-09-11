"""LangGraph agent adapter for CrewAI integration.

This module contains the LangGraphAgentAdapter class that integrates LangGraph ReAct agents
with CrewAI's agent system. Provides memory persistence, tool integration, and structured
output functionality.
"""

from collections.abc import Callable
from typing import Any, cast

from pydantic import ConfigDict, Field, PrivateAttr

from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai.agents.agent_adapters.langgraph.langgraph_tool_adapter import (
    LangGraphToolAdapter,
)
from crewai.agents.agent_adapters.langgraph.protocols import (
    LangGraphCheckPointMemoryModule,
    LangGraphPrebuiltModule,
)
from crewai.agents.agent_adapters.langgraph.structured_output_converter import (
    LangGraphConverterAdapter,
)
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.tools.base_tool import BaseTool
from crewai.utilities import Logger
from crewai.utilities.converter import Converter
from crewai.utilities.import_utils import require


class LangGraphAgentAdapter(BaseAgentAdapter):
    """Adapter for LangGraph agents to work with CrewAI.

    This adapter integrates LangGraph's ReAct agents with CrewAI's agent system,
    providing memory persistence, tool integration, and structured output support.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _logger: Logger = PrivateAttr(default_factory=Logger)
    _tool_adapter: LangGraphToolAdapter = PrivateAttr()
    _graph: Any = PrivateAttr(default=None)
    _memory: Any = PrivateAttr(default=None)
    _max_iterations: int = PrivateAttr(default=10)
    function_calling_llm: Any = Field(default=None)
    step_callback: Callable[..., Any] | None = Field(default=None)

    model: str = Field(default="gpt-4o")
    verbose: bool = Field(default=False)

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: list[BaseTool] | None = None,
        llm: Any = None,
        max_iterations: int = 10,
        agent_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the LangGraph agent adapter.

        Args:
            role: The role description for the agent.
            goal: The primary goal the agent should achieve.
            backstory: Background information about the agent.
            tools: Optional list of tools available to the agent.
            llm: Language model to use, defaults to gpt-4o.
            max_iterations: Maximum number of iterations for task execution.
            agent_config: Additional configuration for the LangGraph agent.
            **kwargs: Additional arguments passed to the base adapter.
        """
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools,
            llm=llm or self.model,
            agent_config=agent_config,
            **kwargs,
        )
        self._tool_adapter = LangGraphToolAdapter(tools=tools)
        self._converter_adapter: LangGraphConverterAdapter = LangGraphConverterAdapter(
            self
        )
        self._max_iterations = max_iterations
        self._setup_graph()

    def _setup_graph(self) -> None:
        """Set up the LangGraph workflow graph.

        Initializes the memory saver and creates a ReAct agent with the configured
        tools, memory checkpointer, and debug settings.
        """

        memory_saver: type[Any] = cast(
            LangGraphCheckPointMemoryModule,
            require(
                "langgraph.checkpoint.memory",
                purpose="LangGraph core functionality",
            ),
        ).MemorySaver
        create_react_agent: Callable[..., Any] = cast(
            LangGraphPrebuiltModule,
            require(
                "langgraph.prebuilt",
                purpose="LangGraph core functionality",
            ),
        ).create_react_agent

        self._memory = memory_saver()

        converted_tools: list[Any] = self._tool_adapter.tools()
        if self._agent_config:
            self._graph = create_react_agent(
                model=self.llm,
                tools=converted_tools,
                checkpointer=self._memory,
                debug=self.verbose,
                **self._agent_config,
            )
        else:
            self._graph = create_react_agent(
                model=self.llm,
                tools=converted_tools or [],
                checkpointer=self._memory,
                debug=self.verbose,
            )

    def _build_system_prompt(self) -> str:
        """Build a system prompt for the LangGraph agent.

        Creates a prompt that includes the agent's role, goal, and backstory,
        then enhances it through the converter adapter for structured output.

        Returns:
            The complete system prompt string.
        """
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
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> str:
        """Execute a task using the LangGraph workflow.

        Configures the agent, processes the task through the LangGraph workflow,
        and handles event emission for execution tracking.

        Args:
            task: The task object to execute.
            context: Optional context information for the task.
            tools: Optional additional tools for this specific execution.

        Returns:
            The final answer from the task execution.

        Raises:
            Exception: If task execution fails.
        """
        self.create_agent_executor(tools)

        self.configure_structured_output(task)

        try:
            task_prompt = task.prompt() if hasattr(task, "prompt") else str(task)

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

            session_id = f"task_{id(task)}"

            config: dict[str, dict[str, str]] = {
                "configurable": {"thread_id": session_id}
            }

            result: dict[str, Any] = self._graph.invoke(
                {
                    "messages": [
                        ("system", self._build_system_prompt()),
                        ("user", task_prompt),
                    ]
                },
                config,
            )

            messages: list[Any] = result.get("messages", [])
            last_message: Any = messages[-1] if messages else None

            final_answer: str = ""
            if isinstance(last_message, dict):
                final_answer = last_message.get("content", "")
            elif hasattr(last_message, "content"):
                final_answer = getattr(last_message, "content", "")

            final_answer = (
                self._converter_adapter.post_process_result(final_answer)
                or "Task execution completed but no clear answer was provided."
            )
            crewai_event_bus.emit(
                self,
                event=AgentExecutionCompletedEvent(
                    agent=self, task=task, output=final_answer
                ),
            )

            return final_answer

        except Exception as e:
            self._logger.log("error", f"Error executing LangGraph task: {e!s}")
            crewai_event_bus.emit(
                self,
                event=AgentExecutionErrorEvent(
                    agent=self,
                    task=task,
                    error=str(e),
                ),
            )
            raise

    def create_agent_executor(self, tools: list[BaseTool] | None = None) -> None:
        """Configure the LangGraph agent for execution.

        Args:
            tools: Optional tools to configure for the agent.
        """
        self.configure_tools(tools)

    def configure_tools(self, tools: list[BaseTool] | None = None) -> None:
        """Configure tools for the LangGraph agent.

        Merges additional tools with existing ones and updates the graph's
        available tools through the tool adapter.

        Args:
            tools: Optional additional tools to configure.
        """
        if tools:
            all_tools: list[BaseTool] = list(self.tools or []) + list(tools or [])
            self._tool_adapter.configure_tools(all_tools)
            available_tools: list[Any] = self._tool_adapter.tools()
            self._graph.tools = available_tools

    def get_delegation_tools(self, agents: list[BaseAgent]) -> list[BaseTool]:
        """Implement delegation tools support for LangGraph.

        Creates delegation tools that allow this agent to delegate tasks to other agents.

        Args:
            agents: List of agents available for delegation.

        Returns:
            List of delegation tools.
        """
        agent_tools: AgentTools = AgentTools(agents=agents)
        return agent_tools.tools()

    @staticmethod
    def get_output_converter(
        llm: Any, text: str, model: Any, instructions: str
    ) -> Converter:
        """Convert output format if needed.

        Args:
            llm: Language model instance.
            text: Text to convert.
            model: Model configuration.
            instructions: Conversion instructions.

        Returns:
            Converter instance for output transformation.
        """
        return Converter(llm=llm, text=text, model=model, instructions=instructions)

    def configure_structured_output(self, task: Any) -> None:
        """Configure the structured output for LangGraph.

        Uses the converter adapter to set up structured output formatting
        based on the task requirements.

        Args:
            task: Task object containing output requirements.
        """
        self._converter_adapter.configure_structured_output(task)
