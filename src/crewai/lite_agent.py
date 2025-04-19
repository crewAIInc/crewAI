import asyncio
import uuid
from datetime import datetime
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
from crewai.llm import LLM
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities import I18N
from crewai.utilities.agent_utils import (
    enforce_rpm_limit,
    format_message_for_llm,
    get_llm_response,
    get_tool_names,
    handle_agent_action_core,
    handle_context_length,
    handle_max_iterations_exceeded,
    handle_output_parser_exception,
    handle_unknown_error,
    has_reached_max_iterations,
    is_context_length_exceeded,
    parse_tools,
    process_llm_response,
    render_text_description_and_args,
    show_agent_logs,
)
from crewai.utilities.converter import convert_to_model, generate_model_description
from crewai.utilities.events.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.printer import Printer
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.tool_utils import execute_tool_and_check_finality


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

    # Core Agent Properties
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Goal of the agent")
    backstory: str = Field(description="Backstory of the agent")
    llm: Optional[Union[str, InstanceOf[LLM], Any]] = Field(
        default=None, description="Language model that will run the agent"
    )
    tools: List[BaseTool] = Field(
        default_factory=list, description="Tools at agent's disposal"
    )

    # Execution Control Properties
    max_iterations: int = Field(
        default=15, description="Maximum number of iterations for tool usage"
    )
    max_execution_time: Optional[int] = Field(
        default=None, description="Maximum execution time in seconds"
    )
    respect_context_window: bool = Field(
        default=True,
        description="Whether to respect the context window of the LLM",
    )
    use_stop_words: bool = Field(
        default=True,
        description="Whether to use stop words to prevent the LLM from using tools",
    )
    request_within_rpm_limit: Optional[Callable[[], bool]] = Field(
        default=None,
        description="Callback to check if the request is within the RPM limit",
    )
    i18n: I18N = Field(default=I18N(), description="Internationalization settings.")

    # Output and Formatting Properties
    response_format: Optional[Type[BaseModel]] = Field(
        default=None, description="Pydantic model for structured output"
    )
    verbose: bool = Field(
        default=False, description="Whether to print execution details"
    )
    callbacks: List[Callable] = Field(
        default=[], description="Callbacks to be used for the agent"
    )

    # State and Results
    tools_results: List[Dict[str, Any]] = Field(
        default=[], description="Results of the tools used by the agent."
    )

    # Reference of Agent
    original_agent: Optional[BaseAgent] = Field(
        default=None, description="Reference to the agent that created this LiteAgent"
    )
    # Private Attributes
    _parsed_tools: List[CrewStructuredTool] = PrivateAttr(default_factory=list)
    _token_process: TokenProcess = PrivateAttr(default_factory=TokenProcess)
    _cache_handler: CacheHandler = PrivateAttr(default_factory=CacheHandler)
    _key: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))
    _messages: List[Dict[str, str]] = PrivateAttr(default_factory=list)
    _iterations: int = PrivateAttr(default=0)
    _printer: Printer = PrivateAttr(default_factory=Printer)
    
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

        return self

    @property
    def key(self) -> str:
        """Get the unique key for this agent instance."""
        return self._key

    @property
    def _original_role(self) -> str:
        """Return the original role for compatibility with tool interfaces."""
        return self.role

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
        # Create agent info for event emission
        agent_info = {
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": self._parsed_tools,
            "verbose": self.verbose,
        }

        try:
            # Reset state for this run
            self._iterations = 0
            self.tools_results = []

            # Format messages for the LLM
            self._messages = self._format_messages(messages)

            # Emit event for agent execution start
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionStartedEvent(
                    agent_info=agent_info,
                    tools=self._parsed_tools,
                    messages=messages,
                ),
            )

            # Execute the agent using invoke loop
            agent_finish = self._invoke_loop()
            formatted_result: Optional[BaseModel] = None
            if self.response_format:
                try:
                    # Cast to BaseModel to ensure type safety
                    result = self.response_format.model_validate_json(
                        agent_finish.output
                    )
                    if isinstance(result, BaseModel):
                        formatted_result = result
                except Exception as e:
                    self._printer.print(
                        content=f"Failed to parse output into response format: {str(e)}",
                        color="yellow",
                    )

            # Calculate token usage metrics
            usage_metrics = self._token_process.get_summary()

            # Create output
            output = LiteAgentOutput(
                raw=agent_finish.output,
                pydantic=formatted_result,
                agent_role=self.role,
                usage_metrics=usage_metrics.model_dump() if usage_metrics else None,
            )

            # Emit completion event
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionCompletedEvent(
                    agent_info=agent_info,
                    output=agent_finish.output,
                ),
            )

            return output

        except Exception as e:
            self._printer.print(
                content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                color="red",
            )
            handle_unknown_error(self._printer, e)
            # Emit error event
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionErrorEvent(
                    agent_info=agent_info,
                    error=str(e),
                ),
            )
            raise e

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
        return await asyncio.to_thread(self.kickoff, messages)

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

    def _invoke_loop(self) -> AgentFinish:
        """
        Run the agent's thought process until it reaches a conclusion or max iterations.

        Returns:
            AgentFinish: The final result of the agent execution.
        """
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
                        callbacks=self._callbacks,
                    )

                enforce_rpm_limit(self.request_within_rpm_limit)

                # Emit LLM call started event
                crewai_event_bus.emit(
                    self,
                    event=LLMCallStartedEvent(
                        messages=self._messages,
                        tools=None,
                        callbacks=self._callbacks,
                    ),
                )

                try:
                    answer = get_llm_response(
                        llm=cast(LLM, self.llm),
                        messages=self._messages,
                        callbacks=self._callbacks,
                        printer=self._printer,
                    )

                    # Emit LLM call completed event
                    crewai_event_bus.emit(
                        self,
                        event=LLMCallCompletedEvent(
                            response=answer,
                            call_type=LLMCallType.LLM_CALL,
                        ),
                    )
                except Exception as e:
                    # Emit LLM call failed event
                    crewai_event_bus.emit(
                        self,
                        event=LLMCallFailedEvent(error=str(e)),
                    )
                    raise e

                formatted_answer = process_llm_response(answer, self.use_stop_words)

                if isinstance(formatted_answer, AgentAction):
                    try:
                        tool_result = execute_tool_and_check_finality(
                            agent_action=formatted_answer,
                            tools=self._parsed_tools,
                            i18n=self.i18n,
                            agent_key=self.key,
                            agent_role=self.role,
                            agent=self.original_agent,
                        )
                    except Exception as e:
                        raise e

                    formatted_answer = handle_agent_action_core(
                        formatted_answer=formatted_answer,
                        tool_result=tool_result,
                        show_logs=self._show_logs,
                    )

                self._append_message(formatted_answer.text, role="assistant")
            except OutputParserException as e:
                formatted_answer = handle_output_parser_exception(
                    e=e,
                    messages=self._messages,
                    iterations=self._iterations,
                    log_error_after=3,
                    printer=self._printer,
                )

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    # Do not retry on litellm errors
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=self._printer,
                        messages=self._messages,
                        llm=cast(LLM, self.llm),
                        callbacks=self._callbacks,
                        i18n=self.i18n,
                    )
                    continue
                else:
                    handle_unknown_error(self._printer, e)
                    raise e

            finally:
                self._iterations += 1

        assert isinstance(formatted_answer, AgentFinish)
        self._show_logs(formatted_answer)
        return formatted_answer

    def _show_logs(self, formatted_answer: Union[AgentAction, AgentFinish]):
        """Show logs for the agent's execution."""
        show_agent_logs(
            printer=self._printer,
            agent_role=self.role,
            formatted_answer=formatted_answer,
            verbose=self.verbose,
        )

    def _append_message(self, text: str, role: str = "assistant") -> None:
        """Append a message to the message list with the given role."""
        self._messages.append(format_message_for_llm(text, role=role))
