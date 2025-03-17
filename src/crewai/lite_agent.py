import asyncio
import json
import re
import uuid  # Add import for generating unique keys
from typing import Any, Dict, List, Optional, Type, Union, cast

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.agents.cache import CacheHandler
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    CrewAgentParser,
    OutputParserException,
)
from crewai.agents.tools_handler import ToolsHandler
from crewai.llm import LLM
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_calling import ToolCalling
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities.events.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.tool_usage_events import ToolUsageStartedEvent
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.token_counter_callback import TokenCalcHandler


class ToolResult:
    """Result of tool execution."""

    def __init__(self, result: str, result_as_answer: bool = False):
        self.result = result
        self.result_as_answer = result_as_answer


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
    _key: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))
    # Store tool results for tracking
    _tools_results: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    # Store messages for conversation
    _messages: List[Dict[str, str]] = PrivateAttr(default_factory=list)
    # Iteration counter
    _iterations: int = PrivateAttr(default=0)
    # Tracking metrics
    _formatting_errors: int = PrivateAttr(default=0)
    _tools_errors: int = PrivateAttr(default=0)
    _delegations: Dict[str, int] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def setup_llm(self):
        """Set up the LLM and other components after initialization."""
        if self.llm is None:
            raise ValueError("LLM must be provided")

        if not isinstance(self.llm, LLM):
            self.llm = create_llm(self.llm)

        return self

    @property
    def key(self) -> str:
        """Get the unique key for this agent instance."""
        return self._key

    @property
    def _original_role(self) -> str:
        """Return the original role for compatibility with tool interfaces."""
        return self.role

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        prompt = f"""You are a helpful AI assistant acting as {self.role}.

Your goal is: {self.goal}

Your backstory: {self.backstory}

When using tools, you MUST follow this EXACT format with the precise spacing and newlines as shown:

Thought: <your reasoning about what needs to be done>

Action: <tool_name>

Action Input: {{
    "parameter1": "value1",
    "parameter2": "value2"
}}

Observation: [Result of the tool execution will appear here]

You can then continue with another tool:

Thought: <your reasoning about what to do next>

Action: <another_tool_name>

Action Input: {{
    "parameter1": "value1"
}}

Observation: [Result of the tool execution will appear here]

When you have a final answer and don't need to use any more tools, respond with:

Thought: <your reasoning about the final answer>

Final Answer: <your final answer to the user>

Here's a concrete example of proper tool usage:

Thought: I need to find out the weather in New York City.

Action: get_weather

Action Input: {{
    "city": "New York City"
}}

Observation: [The weather result would appear here]

Thought: Now I need to save this weather data.

Action: save_weather_data

Action Input: {{
    "filename": "weather_history.txt"
}}

Observation: [The result of saving would appear here]

Thought: I now have all the information I need to answer the user's question.

Final Answer: The weather in New York City today is [weather details] and I've saved this information to the weather_history.txt file.

Always maintain the exact format shown above, with blank lines between sections and properly formatted inputs for tools.
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
                schema_info = ""
                try:
                    if hasattr(tool.args_schema, "model_json_schema"):
                        schema = tool.args_schema.model_json_schema()
                        if "properties" in schema:
                            schema_info = ", ".join(
                                [
                                    f"{k}: {v.get('type', 'any')}"
                                    for k, v in schema["properties"].items()
                                ]
                            )
                        else:
                            schema_info = str(schema)
                except Exception:
                    schema_info = "Unable to parse schema"

                tools_str += f"Parameters: {schema_info}\n"
            tools_str += "\n"

        return tools_str

    def _get_tools_names(self) -> str:
        """Get a comma-separated list of tool names."""
        return ", ".join([tool.name for tool in self.tools])

    def _parse_tools(self) -> List[Dict[str, Any]]:
        """Parse tools to be used by the agent."""
        tools_list = []
        for tool in self.tools:
            try:
                # First try to use the to_structured_tool method if available
                if hasattr(tool, "to_structured_tool"):
                    structured_tool = tool.to_structured_tool()
                    if structured_tool and isinstance(structured_tool, dict):
                        tools_list.append(structured_tool)
                        continue

                # Fall back to manual conversion if to_structured_tool is not available or fails
                tool_dict = {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                    },
                }

                # Add args schema if available
                if hasattr(tool, "args_schema") and tool.args_schema:
                    try:
                        if hasattr(tool.args_schema, "model_json_schema"):
                            tool_dict["function"][
                                "parameters"
                            ] = tool.args_schema.model_json_schema()
                    except Exception as e:
                        if self.verbose:
                            print(
                                f"Warning: Could not get schema for tool {tool.name}: {e}"
                            )

                tools_list.append(tool_dict)

            except Exception as e:
                if self.verbose:
                    print(f"Error converting tool {tool.name}: {e}")

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

    def _preprocess_model_output(self, text: str) -> str:
        """Preprocess the model output to correct common formatting issues."""
        # Skip if the text is empty
        if not text or text.strip() == "":
            return "Thought: I need to provide an answer.\n\nFinal Answer: I don't have enough information to provide a complete answer."

        # Remove 'Action' or 'Final Answer' from anywhere after a proper Thought
        if "Thought:" in text and ("Action:" in text and "Final Answer:" in text):
            # This is a case where both Action and Final Answer appear - clear conflict
            # Check which one appears first and keep only that one
            action_index = text.find("Action:")
            final_answer_index = text.find("Final Answer:")

            if action_index != -1 and final_answer_index != -1:
                if action_index < final_answer_index:
                    # Keep only the Action part
                    text = text[:final_answer_index]
                else:
                    # Keep only the Final Answer part
                    text = text[:action_index] + text[final_answer_index:]

                if self.verbose:
                    print("Removed conflicting Action/Final Answer parts")

        # Check if this looks like a tool usage attempt without proper formatting
        if any(tool.name in text for tool in self.tools) and "Action:" not in text:
            # Try to extract tool name and input
            for tool in self.tools:
                if tool.name in text:
                    # Find the tool name in the text
                    parts = text.split(tool.name, 1)
                    if len(parts) > 1:
                        # Try to extract input as JSON
                        input_text = parts[1]
                        json_match = re.search(r"(\{[\s\S]*\})", input_text)

                        if json_match:
                            # Construct a properly formatted response
                            formatted = "Thought: I need to use a tool to help with this task.\n\n"
                            formatted += f"Action: {tool.name}\n\n"
                            formatted += f"Action Input: {json_match.group(1)}\n"

                            if self.verbose:
                                print(f"Reformatted tool usage: {tool.name}")

                            return formatted

        # Check if this looks like a final answer without proper formatting
        if (
            "Final Answer:" not in text
            and not any(tool.name in text for tool in self.tools)
            and "Action:" not in text
        ):
            # This might be a direct response, format it as a final answer
            # Don't format if text already has a "Thought:" section
            if "Thought:" not in text:
                formatted = "Thought: I can now provide the final answer.\n\n"
                formatted += f"Final Answer: {text}\n"

                if self.verbose:
                    print("Reformatted as final answer")

                return formatted

        return text

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
        # Reset state for this run
        self._iterations = 0
        self._tools_results = []

        # Format messages for the LLM
        self._messages = self._format_messages(messages)

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
            # Execute the agent using invoke loop
            result = await self._invoke()

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

    async def _invoke(self) -> str:
        """
        Run the agent's thought process until it reaches a conclusion or max iterations.
        Similar to _invoke_loop in CrewAgentExecutor.

        Returns:
            str: The final result of the agent execution.
        """
        # Set up tools handler for tool execution
        tools_handler = ToolsHandler(cache=self._cache_handler)

        # Set up callbacks for token tracking
        token_callback = TokenCalcHandler(token_cost_process=self._token_process)
        callbacks = [token_callback]

        # Prepare tool configurations
        parsed_tools = self._parse_tools()
        tools_description = self._format_tools_description()
        tools_names = self._get_tools_names()

        # Create a mapping of tool names to tools for easier lookup
        tool_map = {tool.name: tool for tool in self.tools}

        # Execute the agent loop
        formatted_answer = None
        while self._iterations < self.max_iterations:
            try:
                # Execute the LLM
                llm_instance = self.llm
                if not isinstance(llm_instance, LLM):
                    llm_instance = create_llm(llm_instance)

                if llm_instance is None:
                    raise ValueError(
                        "LLM instance is None. Please provide a valid LLM."
                    )

                # Set response_format if supported
                try:
                    if (
                        self.response_format
                        and hasattr(llm_instance, "response_format")
                        and not llm_instance.response_format
                    ):
                        provider = getattr(
                            llm_instance, "_get_custom_llm_provider", lambda: None
                        )()
                        from litellm.utils import supports_response_schema

                        if hasattr(llm_instance, "model") and supports_response_schema(
                            model=llm_instance.model, custom_llm_provider=provider
                        ):
                            llm_instance.response_format = self.response_format
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not set response_format: {e}")

                # Get the LLM's response
                answer = llm_instance.call(
                    messages=self._messages,
                    tools=parsed_tools,
                    callbacks=callbacks,
                )

                # Keep a copy of the original answer in case we need to fall back to it
                original_answer = answer

                # Pre-process the answer to correct formatting issues
                answer = self._preprocess_model_output(answer)

                # Parse the response into an action or final answer
                parser = CrewAgentParser(agent=cast(BaseAgent, self))
                try:
                    formatted_answer = parser.parse(answer)
                except OutputParserException as e:
                    if self.verbose:
                        print(f"Parser error: {str(e)}")

                    # If we have a Final Answer format error and the original answer is substantive,
                    # return it directly if it looks like a final answer
                    if (
                        "Final Answer" in str(e)
                        and len(original_answer.strip()) > 20
                        and "Action:" not in original_answer
                    ):
                        if self.verbose:
                            print(
                                "Returning original answer directly as final response"
                            )
                        return original_answer

                    # Try to reformat and parse again
                    reformatted = self._preprocess_model_output(
                        "Thought: I need to provide an answer.\n\nFinal Answer: "
                        + original_answer
                    )

                    # Try parsing again
                    try:
                        formatted_answer = parser.parse(reformatted)
                    except Exception:
                        # If we still can't parse, just use the original answer
                        return original_answer

                # If the agent wants to use a tool
                if isinstance(formatted_answer, AgentAction):
                    # Find the appropriate tool
                    tool_name = formatted_answer.tool
                    tool_input = formatted_answer.tool_input

                    # Emit tool usage event
                    crewai_event_bus.emit(
                        self,
                        event=ToolUsageStartedEvent(
                            agent_key=self.key,
                            agent_role=self.role,
                            tool_name=tool_name,
                            tool_args=tool_input,
                            tool_class=tool_name,
                        ),
                    )

                    # Use the tool
                    if tool_name in tool_map:
                        tool = tool_map[tool_name]
                        try:
                            if hasattr(tool, "_run"):
                                # BaseTool interface
                                # Ensure tool_input is a proper dict with string keys
                                if isinstance(tool_input, dict):
                                    result = tool._run(
                                        **{str(k): v for k, v in tool_input.items()}
                                    )
                                else:
                                    result = tool._run(tool_input)
                            elif hasattr(tool, "run"):
                                # Another common interface
                                if isinstance(tool_input, dict):
                                    result = tool.run(
                                        **{str(k): v for k, v in tool_input.items()}
                                    )
                                else:
                                    result = tool.run(tool_input)
                            else:
                                result = f"Error: Tool '{tool_name}' does not have a supported execution method."

                            # Check if tool result should be the final answer
                            result_as_answer = getattr(tool, "result_as_answer", False)

                            # Add to tools_results for tracking
                            self._tools_results.append(
                                {
                                    "result": result,
                                    "tool_name": tool_name,
                                    "tool_args": tool_input,
                                    "result_as_answer": result_as_answer,
                                }
                            )

                            # Create tool result
                            tool_result = ToolResult(
                                result=result, result_as_answer=result_as_answer
                            )

                            # If the tool result should be the final answer, return it
                            if tool_result.result_as_answer:
                                return tool_result.result

                            # Add the result to the formatted answer and messaging
                            formatted_answer.result = tool_result.result
                            formatted_answer.text += (
                                f"\nObservation: {tool_result.result}"
                            )

                            # Execute the step callback if provided
                            if self.step_callback:
                                self.step_callback(formatted_answer)

                            # Add the assistant message to the conversation
                            self._messages.append(
                                {"role": "assistant", "content": formatted_answer.text}
                            )

                        except Exception as e:
                            error_message = f"Error using tool '{tool_name}': {str(e)}"
                            if self.verbose:
                                print(error_message)
                            # Add error message to conversation
                            self._messages.append(
                                {"role": "user", "content": error_message}
                            )
                    else:
                        # Tool not found
                        error_message = f"Tool '{tool_name}' not found. Available tools: {tools_names}"
                        if self.verbose:
                            print(error_message)
                        # Add error message to conversation
                        self._messages.append(
                            {"role": "user", "content": error_message}
                        )

                # If the agent provided a final answer
                elif isinstance(formatted_answer, AgentFinish):
                    # Execute the step callback if provided
                    if self.step_callback:
                        self.step_callback(formatted_answer)

                    # Return the output
                    return formatted_answer.output
                else:
                    # If formatted_answer is None, return the original answer
                    if not formatted_answer and original_answer:
                        return original_answer

                # Increment the iteration counter
                self._iterations += 1

            except Exception as e:
                if self.verbose:
                    print(f"Error during agent execution: {e}")
                # Add error message to conversation
                self._messages.append({"role": "user", "content": f"Error: {str(e)}"})
                self._iterations += 1

        # If we've reached max iterations without a final answer, force one
        if self.verbose:
            print("Maximum iterations reached. Requesting final answer.")

        # Add a message requesting a final answer
        self._messages.append(
            {
                "role": "user",
                "content": "You've been thinking for a while. Please provide your final answer now.",
            }
        )

        # Get the final answer from the LLM
        llm_instance = self.llm
        if not isinstance(llm_instance, LLM):
            llm_instance = create_llm(llm_instance)

        if llm_instance is None:
            raise ValueError("LLM instance is None. Please provide a valid LLM.")

        final_answer = llm_instance.call(
            messages=self._messages,
            callbacks=callbacks,
        )

        return final_answer

    @property
    def tools_results(self) -> List[Dict[str, Any]]:
        """Get the tools results for this agent."""
        return self._tools_results

    def increment_formatting_errors(self) -> None:
        """Increment the formatting errors counter."""
        self._formatting_errors += 1

    def increment_tools_errors(self) -> None:
        """Increment the tools errors counter."""
        self._tools_errors += 1

    def increment_delegations(self, agent_name: Optional[str] = None) -> None:
        """
        Increment the delegations counter for a specific agent.

        Args:
            agent_name: The name of the agent being delegated to.
        """
        if agent_name:
            if agent_name not in self._delegations:
                self._delegations[agent_name] = 0
            self._delegations[agent_name] += 1
