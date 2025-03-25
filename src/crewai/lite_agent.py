import asyncio
import json
import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

from pydantic import BaseModel, Field, InstanceOf, PrivateAttr, model_validator

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.agents.cache import CacheHandler
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from crewai.agents.tools_handler import ToolsHandler
from crewai.llm import LLM
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_usage import ToolUsage, ToolUsageErrorException
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities import I18N
from crewai.utilities.agent_utils import (
    enforce_rpm_limit,
    format_message_for_llm,
    get_llm_response,
    get_tool_names,
    handle_max_iterations_exceeded,
    has_reached_max_iterations,
    parse_tools,
    process_llm_response,
    render_text_description_and_args,
)
from crewai.utilities.converter import convert_to_model, generate_model_description
from crewai.utilities.events.agent_events import (
    LiteAgentExecutionStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
)
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)
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
    llm: Union[str, InstanceOf[LLM], Any] = Field(
        description="Language model that will run the agent"
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
    tools_results: List[Dict[str, Any]] = Field(
        default=[], description="Results of the tools used by the agent."
    )
    respect_context_window: bool = Field(
        default=True,
        description="Whether to respect the context window of the LLM",
    )
    callbacks: List[Callable] = Field(
        default=[], description="Callbacks to be used for the agent"
    )
    _parsed_tools: List[CrewStructuredTool] = PrivateAttr(default_factory=list)
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
    tool_name_to_tool_map: Dict[str, Union[CrewStructuredTool, BaseTool]] = Field(
        default_factory=dict,
        description="Mapping of tool names to tool instances",
    )

    @model_validator(mode="after")
    def setup_llm(self):
        """Set up the LLM and other components after initialization."""
        self.llm = create_llm(self.llm)
        if not isinstance(self.llm, LLM):
            raise ValueError("Unable to create LLM instance")

        # Initialize callbacks
        token_callback = TokenCalcHandler(token_cost_process=self._token_process)
        self._callbacks = [token_callback]

        return self

    @model_validator(mode="after")
    def parse_tools(self):
        """Parse the tools and convert them to CrewStructuredTool instances."""
        self._parsed_tools = parse_tools(self.tools)

        # Initialize tool name to tool mapping
        self.tool_name_to_tool_map = {tool.name: tool for tool in self._parsed_tools}

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
        base_prompt = ""
        if self._parsed_tools:
            # Use the prompt template for agents with tools
            base_prompt = self.i18n.slice("lite_agent_system_prompt_with_tools").format(
                role=self.role,
                backstory=self.backstory,
                goal=self.goal,
                tools=render_text_description_and_args(self._parsed_tools),
                tool_names=get_tool_names(self._parsed_tools),
            )
        else:
            # Use the prompt template for agents without tools
            base_prompt = self.i18n.slice(
                "lite_agent_system_prompt_without_tools"
            ).format(
                role=self.role,
                backstory=self.backstory,
                goal=self.goal,
            )

        # Add response format instructions if specified
        if self.response_format:
            schema = generate_model_description(self.response_format)
            base_prompt += self.i18n.slice("lite_agent_response_format").format(
                response_format=schema
            )

        print("BASE PROMPT:", base_prompt)

        return base_prompt

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
            "tools": self._parsed_tools,
            "verbose": self.verbose,
        }

        # Emit event for agent execution start
        crewai_event_bus.emit(
            self,
            event=LiteAgentExecutionStartedEvent(
                agent_info=agent_info,
                tools=self._parsed_tools,
                messages=messages,
            ),
        )

        try:
            # Execute the agent using invoke loop
            result = await self._invoke()
        except AssertionError:
            self._printer.print(
                content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                color="red",
            )
            raise
        except Exception as e:
            self._handle_unknown_error(e)
            if e.__class__.__module__.startswith("litellm"):
                # Do not retry on litellm errors
                raise e
            else:
                raise e

        formatted_result: Optional[BaseModel] = None
        if self.response_format:
            formatted_result = self.response_format.model_validate_json(result.output)

        return LiteAgentOutput(
            raw=result.output,
            pydantic=formatted_result,
            agent_role=self.role,
            usage_metrics=None,  # TODO: Add usage metrics
        )

    async def _invoke(self) -> AgentFinish:
        """
        Run the agent's thought process until it reaches a conclusion or max iterations.
        Similar to _invoke_loop in CrewAgentExecutor.

        Returns:
            str: The final result of the agent execution.
        """
        # Use the stored callbacks
        callbacks = self._callbacks

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
                        llm=cast(LLM, self.llm),
                        callbacks=callbacks,
                    )

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = get_llm_response(
                    llm=cast(LLM, self.llm),
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

                self._append_message(formatted_answer.text, role="assistant")
            except OutputParserException as e:
                formatted_answer = self._handle_output_parser_exception(e)

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    # Do not retry on litellm errors
                    raise e
                if self._is_context_length_exceeded(e):
                    self._handle_context_length()
                    continue
                else:
                    self._handle_unknown_error(e)
                    raise e

            finally:
                self._iterations += 1

        # During the invoke loop, formatted_answer alternates between AgentAction
        # (when the agent is using tools) and eventually becomes AgentFinish
        # (when the agent reaches a final answer). This assertion confirms we've
        # reached a final answer and helps type checking understand this transition.
        assert isinstance(formatted_answer, AgentFinish)
        self._show_logs(formatted_answer)
        return formatted_answer

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
                agent=self,
                tools=self._parsed_tools,
                action=agent_action,
                tools_handler=None,
                task=None,
                function_calling_llm=None,
            )
            tool_calling = tool_usage.parse_tool_calling(agent_action.text)

            if isinstance(tool_calling, ToolUsageErrorException):
                tool_result = tool_calling.message
                return ToolResult(result=tool_result, result_as_answer=False)
            else:
                if tool_calling.tool_name.casefold().strip() in [
                    tool.name.casefold().strip() for tool in self._parsed_tools
                ] or tool_calling.tool_name.casefold().replace("_", " ") in [
                    tool.name.casefold().strip() for tool in self._parsed_tools
                ]:
                    tool_result = tool_usage.use(tool_calling, agent_action.text)
                    tool = self.tool_name_to_tool_map.get(tool_calling.tool_name)
                    if tool:
                        return ToolResult(
                            result=tool_result, result_as_answer=tool.result_as_answer
                        )
                else:
                    tool_result = self.i18n.errors("wrong_tool_name").format(
                        tool=tool_calling.tool_name,
                        tools=", ".join(
                            [tool.name.casefold() for tool in self._parsed_tools]
                        ),
                    )
                return ToolResult(result=tool_result, result_as_answer=False)

        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=ToolUsageErrorEvent(
                    agent_key=self.key,
                    agent_role=self.role,
                    tool_name=agent_action.tool,
                    tool_args=agent_action.tool_input,
                    tool_class=agent_action.tool,
                    error=str(e),
                ),
            )
            raise e

    def _handle_agent_action(
        self, formatted_answer: AgentAction, tool_result: ToolResult
    ) -> Union[AgentAction, AgentFinish]:
        """Handle the AgentAction, execute tools, and process the results."""

        formatted_answer.text += f"\nObservation: {tool_result.result}"
        formatted_answer.result = tool_result.result

        if tool_result.result_as_answer:
            return AgentFinish(
                thought="",
                output=tool_result.result,
                text=formatted_answer.text,
            )

        self._show_logs(formatted_answer)
        return formatted_answer

    def _show_logs(self, formatted_answer: Union[AgentAction, AgentFinish]):
        if self.verbose:
            agent_role = self.role.split("\n")[0]
            if isinstance(formatted_answer, AgentAction):
                thought = re.sub(r"\n+", "\n", formatted_answer.thought)
                formatted_json = json.dumps(
                    formatted_answer.tool_input,
                    indent=2,
                    ensure_ascii=False,
                )
                self._printer.print(
                    content=f"\n\n\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
                )
                if thought and thought != "":
                    self._printer.print(
                        content=f"\033[95m## Thought:\033[00m \033[92m{thought}\033[00m"
                    )
                self._printer.print(
                    content=f"\033[95m## Using tool:\033[00m \033[92m{formatted_answer.tool}\033[00m"
                )
                self._printer.print(
                    content=f"\033[95m## Tool Input:\033[00m \033[92m\n{formatted_json}\033[00m"
                )
                self._printer.print(
                    content=f"\033[95m## Tool Output:\033[00m \033[92m\n{formatted_answer.result}\033[00m"
                )
            elif isinstance(formatted_answer, AgentFinish):
                self._printer.print(
                    content=f"\n\n\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
                )
                self._printer.print(
                    content=f"\033[95m## Final Answer:\033[00m \033[92m\n{formatted_answer.output}\033[00m\n\n"
                )

    def _append_message(self, text: str, role: str = "assistant") -> None:
        """Append a message to the message list with the given role."""
        self._messages.append(format_message_for_llm(text, role=role))

    def _handle_output_parser_exception(self, e: OutputParserException) -> AgentAction:
        """Handle OutputParserException by updating messages and formatted_answer."""
        self._messages.append({"role": "user", "content": e.error})

        formatted_answer = AgentAction(
            text=e.error,
            tool="",
            tool_input="",
            thought="",
        )

        MAX_ITERATIONS = 3
        if self._iterations > MAX_ITERATIONS:
            self._printer.print(
                content=f"Error parsing LLM output, agent will retry: {e.error}",
                color="red",
            )

        return formatted_answer

    def _is_context_length_exceeded(self, exception: Exception) -> bool:
        """Check if the exception is due to context length exceeding."""
        return LLMContextLengthExceededException(
            str(exception)
        )._is_context_limit_error(str(exception))

    def _handle_context_length(self) -> None:
        if self.respect_context_window:
            self._printer.print(
                content="Context length exceeded. Summarizing content to fit the model context window.",
                color="yellow",
            )
            self._summarize_messages()
        else:
            self._printer.print(
                content="Context length exceeded. Consider using smaller text or RAG tools from crewai_tools.",
                color="red",
            )
            raise SystemExit(
                "Context length exceeded and user opted not to summarize. Consider using smaller text or RAG tools from crewai_tools."
            )

    def _summarize_messages(self) -> None:
        messages_groups = []
        for message in self.messages:
            content = message["content"]
            cut_size = cast(LLM, self.llm).get_context_window_size()
            for i in range(0, len(content), cut_size):
                messages_groups.append(content[i : i + cut_size])

        summarized_contents = []
        for group in messages_groups:
            summary = cast(LLM, self.llm).call(
                [
                    format_message_for_llm(
                        self.i18n.slice("summarizer_system_message"), role="system"
                    ),
                    format_message_for_llm(
                        self.i18n.slice("summarize_instruction").format(group=group),
                    ),
                ],
                callbacks=self.callbacks,
            )
            summarized_contents.append(summary)

        merged_summary = " ".join(str(content) for content in summarized_contents)

        self.messages = [
            format_message_for_llm(
                self.i18n.slice("summary").format(merged_summary=merged_summary)
            )
        ]

    def _handle_unknown_error(self, exception: Exception) -> None:
        """Handle unknown errors by informing the user."""
        self._printer.print(
            content="An unknown error occurred. Please check the details below.",
            color="red",
        )
        self._printer.print(
            content=f"Error details: {exception}",
            color="red",
        )
