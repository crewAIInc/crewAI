from typing import Any, AsyncIterable, Dict, List, Optional

from pydantic import Field, PrivateAttr

from crewai.agents.agent_adapters.base_agent_adapter import BaseAgentAdapter
from crewai.agents.agent_adapters.langgraph.langgraph_tool_adapter import (
    LangGraphToolAdapter,
)
from crewai.agents.agent_adapters.langgraph.structured_output_converter import (
    LangGraphConverterAdapter,
)
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.tools.base_tool import BaseTool
from crewai.utilities import Logger
from crewai.utilities.converter import Converter
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)

try:
    from langchain_core.messages import ToolMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class LangGraphAgentAdapter(BaseAgentAdapter):
    """Adapter for LangGraph agents to work with CrewAI."""

    model_config = {"arbitrary_types_allowed": True}

    _logger: Logger = PrivateAttr(default_factory=lambda: Logger())
    _tool_adapter: LangGraphToolAdapter = PrivateAttr()
    _graph: Any = PrivateAttr(default=None)
    _memory: Any = PrivateAttr(default=None)
    _max_iterations: int = PrivateAttr(default=10)
    function_calling_llm: Any = Field(default=None)
    step_callback: Any = Field(default=None)

    model: str = Field(default="gpt-4o")
    verbose: bool = Field(default=False)

    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: Optional[List[BaseTool]] = None,
        llm: Any = None,
        max_iterations: int = 10,
        agent_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the LangGraph agent adapter."""
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph Agent Dependencies are not installed. Please install it using `uv add langchain-core langgraph`"
            )
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
        self._converter_adapter = LangGraphConverterAdapter(self)
        self._max_iterations = max_iterations
        self._setup_graph()

    def _setup_graph(self) -> None:
        """Set up the LangGraph workflow graph."""
        try:
            self._memory = MemorySaver()

            converted_tools: List[Any] = self._tool_adapter.tools()
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

        except ImportError as e:
            self._logger.log(
                "error", f"Failed to import LangGraph dependencies: {str(e)}"
            )
            raise
        except Exception as e:
            self._logger.log("error", f"Error setting up LangGraph agent: {str(e)}")
            raise

    def _build_system_prompt(self) -> str:
        """Build a system prompt for the LangGraph agent."""
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
        """Execute a task using the LangGraph workflow."""
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

            config = {"configurable": {"thread_id": session_id}}

            result = self._graph.invoke(
                {
                    "messages": [
                        ("system", self._build_system_prompt()),
                        ("user", task_prompt),
                    ]
                },
                config,
            )

            messages = result.get("messages", [])
            last_message = messages[-1] if messages else None

            final_answer = ""
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
            self._logger.log("error", f"Error executing LangGraph task: {str(e)}")
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
        """Configure the LangGraph agent for execution."""
        self.configure_tools(tools)

    def configure_tools(self, tools: Optional[List[BaseTool]] = None) -> None:
        """Configure tools for the LangGraph agent."""
        if tools:
            all_tools = list(self.tools or []) + list(tools or [])
            self._tool_adapter.configure_tools(all_tools)
            available_tools = self._tool_adapter.tools()
            self._graph.tools = available_tools

    def get_delegation_tools(self, agents: List[BaseAgent]) -> List[BaseTool]:
        """Implement delegation tools support for LangGraph."""
        agent_tools = AgentTools(agents=agents)
        return agent_tools.tools()

    def get_output_converter(
        self, llm: Any, text: str, model: Any, instructions: str
    ) -> Any:
        """Convert output format if needed."""
        return Converter(llm=llm, text=text, model=model, instructions=instructions)

    def configure_structured_output(self, task) -> None:
        """Configure the structured output for LangGraph."""
        self._converter_adapter.configure_structured_output(task)
