from typing import Any, Dict, List, Optional

from agents import Agent as OpenAIAgent
from agents import FunctionTool, Runner, Tool, enable_verbose_stdout_logging
from pydantic import BaseModel, Field, PrivateAttr

from crewai.agent import BaseAgent
from crewai.tools import BaseTool

from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities import Logger
from crewai.utilities.events import crewai_event_bus
from crewai.utilities.events.agent_events import (
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)


class OpenAIAgentAdapter(BaseAgent, BaseModel):
    """Adapter for OpenAI Assistants"""

    model_config = {"arbitrary_types_allowed": True}

    _openai_agent: OpenAIAgent = PrivateAttr()
    _logger: Logger = PrivateAttr(default_factory=lambda: Logger())
    _active_thread: Optional[str] = PrivateAttr(default=None)
    function_calling_llm: Any = Field(default=None)
    step_callback: Any = Field(default=None)
    converted_tools: Optional[List[Tool]] = Field(default=None)

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

    def execute_task(
        self,
        task: Any,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """Execute a task using the OpenAI Assistant"""
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
            # This is pretty much the agent_executor logic:
            result = self.agent_executor.run_sync(self._openai_agent, task_prompt)
            return result.final_output

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
            self._configure_tools(all_tools)
            if self.converted_tools:
                self._openai_agent.tools = self.converted_tools

        self.agent_executor = Runner

    def _prepare_task_input(self, task: Any, context: Optional[str]) -> str:
        """Prepare the task input with context if available"""
        task_input = task.description if hasattr(task, "description") else str(task)
        if context:
            task_input = f"Context:\n{context}\n\nTask:\n{task_input}"
        return task_input

    def _configure_tools(self, tools: List[BaseTool]) -> None:
        """Configure tools for the OpenAI Assistant"""
        if self.tools:
            all_tools = tools + self.tools
        else:
            all_tools = tools
        openai_tools = self._convert_tools_to_openai_format(all_tools)
        self.converted_tools = openai_tools

    def _convert_tools_to_openai_format(
        self, tools: Optional[List[BaseTool]]
    ) -> List[Tool]:
        """Convert CrewAI tools to OpenAI Assistant tool format"""
        if not tools:
            return []

        def sanitize_tool_name(name: str) -> str:
            """Convert tool name to match OpenAI's required pattern"""
            import re

            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name).lower()
            return sanitized

        def create_tool_wrapper(tool: BaseTool):
            """Create a wrapper function that handles the OpenAI function tool interface"""

            async def wrapper(context_wrapper: Any, arguments: Any) -> Any:
                # Get the parameter name from the schema
                param_name = list(
                    tool.args_schema.model_json_schema()["properties"].keys()
                )[0]

                # Handle different argument types
                if isinstance(arguments, dict):
                    args_dict = arguments
                elif isinstance(arguments, str):
                    try:
                        import json

                        args_dict = json.loads(arguments)
                    except json.JSONDecodeError:
                        args_dict = {param_name: arguments}
                else:
                    args_dict = {param_name: str(arguments)}

                # Run the tool with the processed arguments
                result = tool._run(**args_dict)

                # Ensure the result is JSON serializable
                if isinstance(result, (dict, list, str, int, float, bool, type(None))):
                    return result
                return str(result)

            return wrapper

        openai_tools = []
        for tool in tools:
            print("tools coming in", tool)
            schema = tool.args_schema.model_json_schema()

            schema.update({"additionalProperties": False, "type": "object"})
            print("schema here", schema)

            openai_tool = FunctionTool(
                name=sanitize_tool_name(tool.name),
                description=tool.description,
                params_json_schema=schema,
                on_invoke_tool=create_tool_wrapper(tool),
            )
            openai_tools.append(openai_tool)

        return openai_tools

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
        return tools
