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
        tools: Optional[List[BaseTool]] = [],
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
        self._tool_adapter = LangGraphToolAdapter(tools=tools or [])
        self._converter_adapter = LangGraphConverterAdapter(self)
        self._max_iterations = max_iterations
        self._setup_graph()

    def _setup_graph(self) -> None:
        """Set up the LangGraph workflow graph."""
        try:
            # Initialize memory for the agent
            self._memory = MemorySaver()

            converted_tools = self._tool_adapter.converted_tools

            self._graph = create_react_agent(
                model=self.llm,
                tools=converted_tools,
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

            # Set up a session ID for this task
            session_id = f"task_{id(task)}"

            # Configure the invocation
            config = {"configurable": {"thread_id": session_id}}

            # Invoke the agent graph with the task prompt
            result = self._graph.invoke(
                {
                    "messages": [
                        ("system", self._build_system_prompt()),
                        ("user", task_prompt),
                    ]
                },
                config,
            )

            # Get the final response
            messages = result.get("messages", [])
            last_message = messages[-1] if messages else None

            final_answer = ""
            if isinstance(last_message, dict):
                final_answer = last_message.get("content", "")
            elif hasattr(last_message, "content"):
                final_answer = getattr(last_message, "content", "")

            # Post-process to ensure correct structured output format if needed
            final_answer = (
                self._converter_adapter.post_process_result(final_answer)
                or "Task execution completed but no clear answer was provided."
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

    async def stream_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> AsyncIterable[Dict[str, Any]]:
        """Stream the execution of a task."""
        self.create_agent_executor(tools)

        try:
            task_prompt = task.prompt() if hasattr(task, "prompt") else str(task)

            if context:
                task_prompt = self.i18n.slice("task_with_context").format(
                    task=task_prompt, context=context
                )

            # Set up a session ID for this task
            session_id = f"task_{id(task)}"

            # Configure the invocation
            config = {"configurable": {"thread_id": session_id}}

            # Stream the execution
            inputs = {"messages": [("user", task_prompt)]}

            for item in self._graph.stream(inputs, config, stream_mode="values"):
                message = item.get("messages", [])[-1] if "messages" in item else None

                if (
                    message is not None
                    and hasattr(message, "tool_calls")
                    and getattr(message, "tool_calls", None)
                ):
                    tool_calls = getattr(message, "tool_calls", [])
                    if tool_calls and len(tool_calls) > 0:
                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": f"Using tool: {tool_calls[0].name}",
                        }
                elif isinstance(message, ToolMessage):
                    content = getattr(message, "content", "Tool execution complete")
                    yield {
                        "is_task_complete": False,
                        "require_user_input": False,
                        "content": f"Tool result: {content[:50]}...",
                    }
                elif message is not None:
                    # Final response or intermediary thinking
                    content = getattr(message, "content", str(message))
                    yield {
                        "is_task_complete": True,
                        "require_user_input": False,
                        "content": content,
                    }

        except Exception as e:
            self._logger.log("error", f"Error streaming LangGraph task: {str(e)}")
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Error: {str(e)}",
            }

    def create_agent_executor(self, tools: Optional[List[BaseTool]] = None) -> None:
        """Configure the LangGraph agent for execution."""
        if tools:
            self.configure_tools(tools)

        # No need for a separate executor in LangGraph

    def configure_tools(self, tools: Optional[List[BaseTool]] = None) -> None:
        """Configure tools for the LangGraph agent."""
        if tools:
            all_tools = list(self.tools or []) + list(tools or [])
            self._tool_adapter.configure_tools(all_tools)
            # We need to recreate the graph with the new tools
            self._setup_graph()

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
