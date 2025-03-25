import asyncio
import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

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
from crewai.tools.tool_usage import ToolUsage, ToolUsageErrorException
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities import I18N
from crewai.utilities.agent_utils import (
    enforce_rpm_limit,
    get_llm_response,
    get_tool_names,
    handle_max_iterations_exceeded,
    has_reached_max_iterations,
    parse_tools,
    process_llm_response,
    render_text_description_and_args,
)
from crewai.utilities.events.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.tool_usage_events import ToolUsageStartedEvent
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.printer import Printer
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
    """

    model_config = {"arbitrary_types_allowed": True}

    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Goal of the agent")
    backstory: str = Field(description="Backstory of the agent")
    llm: LLM = Field(description="Language model that will run the agent")
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

    step_callback: Optional[Any] = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    tools_results: Optional[List[Dict[str, Any]]] = Field(
        default=[], description="Results of the tools used by the agent."
    )

    _token_process: TokenProcess = PrivateAttr(default_factory=TokenProcess)
    _cache_handler: CacheHandler = PrivateAttr(default_factory=CacheHandler)
    _times_executed: int = PrivateAttr(default=0)
    _max_retry_limit: int = PrivateAttr(default=2)
    _key: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))
    # Store messages for conversation
    _messages: List[Dict[str, str]] = PrivateAttr(default_factory=list)
    # Iteration counter
    _iterations: int = PrivateAttr(default=0)
    # Tracking metrics
    _formatting_errors: int = PrivateAttr(default=0)
    _tools_errors: int = PrivateAttr(default=0)
    _delegations: Dict[str, int] = PrivateAttr(default_factory=dict)
    # Internationalization
    _printer: Printer = PrivateAttr(default_factory=Printer)
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")
    request_within_rpm_limit: Optional[Callable[[], bool]] = Field(
        default=None,
        description="Callback to check if the request is within the RPM limit",
    )
    use_stop_words: bool = Field(
        default=True,
        description="Whether to use stop words to prevent the LLM from using tools",
    )

    @model_validator(mode="after")
    def setup_llm(self):
        """Set up the LLM and other components after initialization."""
        if self.llm is None:
            raise ValueError("LLM must be provided")

        if not isinstance(self.llm, LLM):
            self.llm = create_llm(self.llm)
            self.use_stop_words = self.llm.supports_stop_words()

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
        if self.tools:
            # Use the prompt template for agents with tools
            return self.i18n.slice("lite_agent_system_prompt_with_tools").format(
                role=self.role,
                backstory=self.backstory,
                goal=self.goal,
                tools=render_text_description_and_args(self.tools),
                tool_names=get_tool_names(self.tools),
            )
        else:
            # Use the prompt template for agents without tools
            return self.i18n.slice("lite_agent_system_prompt_without_tools").format(
                role=self.role,
                backstory=self.backstory,
                goal=self.goal,
            )

    def _format_messages(
        self, messages: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Format messages for the LLM."""
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        system_prompt = self._get_default_system_prompt()

        # Add system message at the beginning
        formatted_messages = [{"role": "system", "content": system_prompt}]

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
        self.tools_results = []

        # Format messages for the LLM
        self._messages = self._format_messages(messages)

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
                messages=messages,
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
        # # Set up tools handler for tool execution
        # tools_handler = ToolsHandler(cache=self._cache_handler)

        # TODO: MOVE TO INIT
        # Set up callbacks for token tracking
        token_callback = TokenCalcHandler(token_cost_process=self._token_process)
        callbacks = [token_callback]

        # # Prepare tool configurations
        # parsed_tools = parse_tools(self.tools)
        # tools_description = render_text_description_and_args(parsed_tools)
        # tools_names = get_tool_names(parsed_tools)

        # Execute the agent loop
        formatted_answer = None
        while not isinstance(formatted_answer, AgentFinish):
            try:
                if has_reached_max_iterations(self._iterations, self.max_iterations):
                    formatted_answer = handle_max_iterations_exceeded(
                        formatted_answer,
                        printer=self._printer,
                        i18n=self.i18n,
                        messages=self._messages,
                        llm=self.llm,
                        callbacks=callbacks,
                    )

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = get_llm_response(
                    llm=self.llm,
                    messages=self._messages,
                    callbacks=callbacks,
                    printer=self._printer,
                )
                formatted_answer = process_llm_response(answer, self.use_stop_words)

                if isinstance(formatted_answer, AgentAction):
                    tool_result = self._execute_tool_and_check_finality(
                        formatted_answer
                    )
                    formatted_answer = self._handle_agent_action(
                        formatted_answer, tool_result
                    )

                self._invoke_step_callback(formatted_answer)
                self._append_message(formatted_answer.text, role="assistant")

            except Exception as e:
                print(f"Error: {e}")

    def _execute_tool_and_check_finality(self, agent_action: AgentAction) -> ToolResult:
        try:
            crewai_event_bus.emit(
                self,
                event=ToolUsageStartedEvent(
                    agent_key=self.key,
                    agent_role=self.role,
                    tool_name=agent_action.tool,
                    tool_args=agent_action.tool_input,
                    tool_class=agent_action.tool,
                ),
            )
            tool_usage = ToolUsage(
                tools=self.tools,
                original_tools=self.tools,  # TODO: INVESTIGATE DIFF BETWEEN THIS AND ABOVE
                tools_description=render_text_description_and_args(self.tools),
                tools_names=get_tool_names(self.tools),
                agent=self,
                action=agent_action,
            )
            tool_calling = tool_usage.parse_tool_calling(agent_action.text)

            if isinstance(tool_calling, ToolUsageErrorException):
                tool_result = tool_calling.message
                return ToolResult(result=tool_result, result_as_answer=False)
            else:
                if tool_calling.tool_name.casefold().strip() in [
                    name.casefold().strip() for name in self.tool_name_to_tool_map
                ] or tool_calling.tool_name.casefold().replace("_", " ") in [
                    name.casefold().strip() for name in self.tool_name_to_tool_map
                ]:
                    tool_result = tool_usage.use(tool_calling, agent_action.text)
                    tool = self.tool_name_to_tool_map.get(tool_calling.tool_name)
                    if tool:
                        return ToolResult(
                            result=tool_result, result_as_answer=tool.result_as_answer
                        )
                else:
                    tool_result = self._i18n.errors("wrong_tool_name").format(
                        tool=tool_calling.tool_name,
                        tools=", ".join([tool.name.casefold() for tool in self.tools]),
                    )
                return ToolResult(result=tool_result, result_as_answer=False)

        except Exception as e:
            if self.agent:
                crewai_event_bus.emit(
                    self,
                    event=ToolUsageErrorEvent(  # validation error
                        agent_key=self.agent.key,
                        agent_role=self.agent.role,
                        tool_name=agent_action.tool,
                        tool_args=agent_action.tool_input,
                        tool_class=agent_action.tool,
                        error=str(e),
                    ),
                )
            raise e
