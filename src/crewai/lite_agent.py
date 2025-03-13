import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Type, Union, cast

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.agents.cache import CacheHandler
from crewai.llm import LLM
from crewai.tools.base_tool import BaseTool
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities.events.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.token_counter_callback import TokenCalcHandler


class LiteAgentOutput(BaseModel):
    """Class that represents the result of a LiteAgent execution."""

    model_config = {"arbitrary_types_allowed": True}

    raw: str = Field(description="Raw output of the agent", default="")
    pydantic: Optional[BaseModel] = Field(
        description="Pydantic output of the agent", default=None
    )
    agent_role: str = Field(description="Role of the agent that produced this output")
    usage_metrics: Optional[Dict[str, Any]] = Field(
        description="Token usage metrics for this execution", default=None
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert pydantic_output to a dictionary."""
        if self.pydantic:
            return self.pydantic.model_dump()
        return {}

    def __str__(self) -> str:
        """String representation of the output."""
        if self.pydantic:
            return str(self.pydantic)
        return self.raw


class LiteAgent(BaseModel):
    """
    A lightweight agent that can process messages and use tools.

    This agent is simpler than the full Agent class, focusing on direct execution
    rather than task delegation. It's designed to be used for simple interactions
    where a full crew is not needed.

    Attributes:
        role: The role of the agent.
        goal: The objective of the agent.
        backstory: The backstory of the agent.
        llm: The language model that will run the agent.
        tools: Tools at the agent's disposal.
        verbose: Whether the agent execution should be in verbose mode.
        max_iterations: Maximum number of iterations for tool usage.
        max_execution_time: Maximum execution time in seconds.
        response_format: Optional Pydantic model for structured output.
        system_prompt: Custom system prompt to override the default.
    """

    model_config = {"arbitrary_types_allowed": True}

    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Goal of the agent")
    backstory: str = Field(description="Backstory of the agent")
    llm: Union[str, LLM, Any] = Field(
        description="Language model that will run the agent", default=None
    )
    tools: List[BaseTool] = Field(
        default_factory=list, description="Tools at agent's disposal"
    )
    verbose: bool = Field(
        default=False, description="Whether to print execution details"
    )
    max_iterations: int = Field(
        default=15, description="Maximum number of iterations for tool usage"
    )
    max_execution_time: Optional[int] = Field(
        default=None, description="Maximum execution time in seconds"
    )
    response_format: Optional[Type[BaseModel]] = Field(
        default=None, description="Pydantic model for structured output"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="Custom system prompt to override default"
    )
    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )

    _token_process: TokenProcess = PrivateAttr(default_factory=TokenProcess)
    _cache_handler: CacheHandler = PrivateAttr(default_factory=CacheHandler)
    _times_executed: int = PrivateAttr(default=0)
    _max_retry_limit: int = PrivateAttr(default=2)

    @model_validator(mode="after")
    def setup_llm(self):
        """Set up the LLM after initialization."""
        if self.llm is None:
            raise ValueError("LLM must be provided")

        if not isinstance(self.llm, LLM):
            self.llm = create_llm(self.llm)

        return self

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        prompt = f"""You are a helpful AI assistant acting as {self.role}.

Your goal is: {self.goal}

Your backstory: {self.backstory}

When using tools, follow this format:
Thought: I need to use a tool to help with this task.
Action: tool_name
Action Input: {{
    "parameter1": "value1",
    "parameter2": "value2"
}}
Observation: [Result of the tool execution]

When you have the final answer, respond directly without the above format.
"""
        return prompt

    def _format_tools_description(self) -> str:
        """Format tools into a string for the prompt."""
        if not self.tools:
            return "You don't have any tools available."

        tools_str = "You have access to the following tools:\n\n"
        for tool in self.tools:
            tools_str += f"Tool: {tool.name}\n"
            tools_str += f"Description: {tool.description}\n"
            if hasattr(tool, "args_schema"):
                tools_str += f"Parameters: {tool.args_schema}\n"
            tools_str += "\n"

        return tools_str

    def _parse_tools(self) -> List[Any]:
        """Parse tools to be used by the agent."""
        tools_list = []
        try:
            from crewai.tools import BaseTool as CrewAITool

            for tool in self.tools:
                if isinstance(tool, CrewAITool):
                    tools_list.append(tool.to_structured_tool())
                else:
                    tools_list.append(tool)
        except ModuleNotFoundError:
            tools_list = self.tools

        return tools_list

    def _format_messages(
        self, messages: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Format messages for the LLM."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        system_prompt = self.system_prompt or self._get_default_system_prompt()
        tools_description = self._format_tools_description()

        # Add system message at the beginning
        formatted_messages = [
            {"role": "system", "content": f"{system_prompt}\n\n{tools_description}"}
        ]

        # Add the rest of the messages
        formatted_messages.extend(messages)

        return formatted_messages

    def _extract_structured_output(self, text: str) -> Optional[BaseModel]:
        """Extract structured output from text if response_format is set."""
        if not self.response_format:
            return None

        try:
            # Try to extract JSON from the text
            json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
            if json_match:
                json_str = json_match.group(1)
                json_data = json.loads(json_str)
            else:
                # Try to parse the entire text as JSON
                try:
                    json_data = json.loads(text)
                except json.JSONDecodeError:
                    # If that fails, use a more lenient approach to find JSON-like content
                    potential_json = re.search(r"(\{[\s\S]*\})", text)
                    if potential_json:
                        json_data = json.loads(potential_json.group(1))
                    else:
                        return None

            # Convert to Pydantic model
            return self.response_format.model_validate(json_data)
        except Exception as e:
            if self.verbose:
                print(f"Error extracting structured output: {e}")
            return None

    def kickoff(self, messages: Union[str, List[Dict[str, str]]]) -> LiteAgentOutput:
        """
        Execute the agent with the given messages.

        Args:
            messages: Either a string query or a list of message dictionaries.
                     If a string is provided, it will be converted to a user message.
                     If a list is provided, each dict should have 'role' and 'content' keys.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        return asyncio.run(self.kickoff_async(messages))

    async def kickoff_async(
        self, messages: Union[str, List[Dict[str, str]]]
    ) -> LiteAgentOutput:
        """
        Execute the agent asynchronously with the given messages.

        Args:
            messages: Either a string query or a list of message dictionaries.
                     If a string is provided, it will be converted to a user message.
                     If a list is provided, each dict should have 'role' and 'content' keys.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        # Format messages for the LLM
        formatted_messages = self._format_messages(messages)

        # Prepare tools
        parsed_tools = self._parse_tools()

        # Get the original query for event emission
        query = messages if isinstance(messages, str) else messages[-1]["content"]

        # Create agent info for event emission
        agent_info = {
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": self.tools,
            "verbose": self.verbose,
        }

        # Emit event for agent execution start
        crewai_event_bus.emit(
            self,
            event=LiteAgentExecutionStartedEvent(
                agent_info=agent_info,
                tools=self.tools,
                task_prompt=query,
            ),
        )

        try:
            # Execute the agent
            result = await self._execute_agent(formatted_messages, parsed_tools)

            # Extract structured output if response_format is set
            pydantic_output = None

            if self.response_format:
                structured_output = self._extract_structured_output(result)
                if isinstance(structured_output, BaseModel):
                    pydantic_output = structured_output

            # Create output object
            usage_metrics = {}
            if hasattr(self._token_process, "get_summary"):
                usage_metrics_obj = self._token_process.get_summary()
                if isinstance(usage_metrics_obj, UsageMetrics):
                    usage_metrics = usage_metrics_obj.model_dump()

            output = LiteAgentOutput(
                raw=result,
                pydantic=pydantic_output,
                agent_role=self.role,
                usage_metrics=usage_metrics,
            )

            # Emit event for agent execution completion
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionCompletedEvent(
                    agent_info=agent_info,
                    output=result,
                ),
            )

            return output

        except Exception as e:
            # Emit event for agent execution error
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionErrorEvent(
                    agent_info=agent_info,
                    error=str(e),
                ),
            )

            # Retry if we haven't exceeded the retry limit
            self._times_executed += 1
            if self._times_executed <= self._max_retry_limit:
                if self.verbose:
                    print(
                        f"Retrying agent execution ({self._times_executed}/{self._max_retry_limit})..."
                    )
                return await self.kickoff_async(messages)

            raise e

    async def _execute_agent(
        self, messages: List[Dict[str, str]], tools: List[Any]
    ) -> str:
        """
        Execute the agent with the given messages and tools.

        Args:
            messages: List of message dictionaries.
            tools: List of parsed tools.

        Returns:
            str: The result of the agent execution.
        """
        # Set up available functions for tool execution
        available_functions = {}
        for tool in self.tools:
            available_functions[tool.name] = tool.run

        # Set up callbacks for token tracking
        token_callback = TokenCalcHandler(token_cost_process=self._token_process)
        callbacks = [token_callback]

        # Execute the LLM with the messages and tools
        llm_instance = self.llm
        if not isinstance(llm_instance, LLM):
            llm_instance = create_llm(llm_instance)

        if llm_instance is None:
            raise ValueError("LLM instance is None. Please provide a valid LLM.")

        # Set the response_format on the LLM instance if it's not already set
        if self.response_format and not llm_instance.response_format:
            llm_instance.response_format = self.response_format

        # Convert tools to dictionaries for LLM call
        formatted_tools = None
        if tools:
            formatted_tools = []
            for tool in tools:
                if hasattr(tool, "dict"):
                    formatted_tools.append(tool.dict())
                elif hasattr(tool, "to_dict"):
                    formatted_tools.append(tool.to_dict())
                elif hasattr(tool, "model_dump"):
                    formatted_tools.append(tool.model_dump())
                else:
                    # If we can't convert the tool, skip it
                    if self.verbose:
                        print(
                            f"Warning: Could not convert tool {tool} to dictionary format"
                        )

        result = llm_instance.call(
            messages=messages,
            tools=formatted_tools,
            callbacks=callbacks,
            available_functions=available_functions,
        )

        return result
