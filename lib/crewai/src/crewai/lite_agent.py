from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
import inspect
import json
from types import MethodType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    get_args,
    get_origin,
)
import uuid
import warnings

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    InstanceOf,
    PrivateAttr,
    field_validator,
    model_validator,
)
from typing_extensions import Self


if TYPE_CHECKING:
    from crewai_files import FileInput

    from crewai.a2a.config import A2AClientConfig, A2AConfig, A2AServerConfig

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
)
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.logging_events import AgentLogsExecutionEvent
from crewai.flow.flow_trackable import FlowTrackable
from crewai.hooks.llm_hooks import get_after_llm_call_hooks, get_before_llm_call_hooks
from crewai.hooks.types import AfterLLMCallHookType, BeforeLLMCallHookType
from crewai.lite_agent_output import LiteAgentOutput
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
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
)
from crewai.utilities.converter import (
    Converter,
    ConverterError,
)
from crewai.utilities.guardrail import process_guardrail
from crewai.utilities.guardrail_types import GuardrailCallable, GuardrailType
from crewai.utilities.i18n import I18N, get_i18n
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.printer import Printer
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from crewai.utilities.types import LLMMessage


def _kickoff_with_a2a_support(
    agent: LiteAgent,
    original_kickoff: Callable[..., LiteAgentOutput],
    messages: str | list[LLMMessage],
    response_format: type[BaseModel] | None,
    input_files: dict[str, FileInput] | None,
    extension_registry: Any,
) -> LiteAgentOutput:
    """Wrap kickoff with A2A delegation using Task adapter.

    Args:
        agent: The LiteAgent instance.
        original_kickoff: The original kickoff method.
        messages: Input messages.
        response_format: Optional response format.
        input_files: Optional input files.
        extension_registry: A2A extension registry.

    Returns:
        LiteAgentOutput from either local execution or A2A delegation.
    """
    from crewai.a2a.utils.response_model import get_a2a_agents_and_response_model
    from crewai.a2a.wrapper import _execute_task_with_a2a
    from crewai.task import Task

    a2a_agents, agent_response_model = get_a2a_agents_and_response_model(agent.a2a)

    if not a2a_agents:
        return original_kickoff(messages, response_format, input_files)

    if isinstance(messages, str):
        description = messages
    else:
        content = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            None,
        )
        description = content if isinstance(content, str) else ""

    if not description:
        return original_kickoff(messages, response_format, input_files)

    fake_task = Task(
        description=description,
        agent=agent,
        expected_output="Result from A2A delegation",
        input_files=input_files or {},
    )

    def task_to_kickoff_adapter(
        self: Any, task: Task, context: str | None, tools: list[Any] | None
    ) -> str:
        result = original_kickoff(messages, response_format, input_files)
        return result.raw

    result_str = _execute_task_with_a2a(
        self=agent,  # type: ignore[arg-type]
        a2a_agents=a2a_agents,
        original_fn=task_to_kickoff_adapter,
        task=fake_task,
        agent_response_model=agent_response_model,
        context=None,
        tools=None,
        extension_registry=extension_registry,
    )

    return LiteAgentOutput(
        raw=result_str,
        pydantic=None,
        agent_role=agent.role,
        usage_metrics=None,
        messages=[],
    )


class LiteAgent(FlowTrackable, BaseModel):
    """
    A lightweight agent that can process messages and use tools.

    .. deprecated::
        LiteAgent is deprecated and will be removed in a future version.
        Use ``Agent().kickoff(messages)`` instead, which provides the same
        functionality with additional features like memory and knowledge support.

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
    id: UUID4 = Field(default_factory=uuid.uuid4, frozen=True)
    role: str = Field(description="Role of the agent")
    goal: str = Field(description="Goal of the agent")
    backstory: str = Field(description="Backstory of the agent")
    llm: str | InstanceOf[BaseLLM] | Any | None = Field(
        default=None, description="Language model that will run the agent"
    )
    tools: list[BaseTool] = Field(
        default_factory=list, description="Tools at agent's disposal"
    )
    max_iterations: int = Field(
        default=15, description="Maximum number of iterations for tool usage"
    )
    max_execution_time: int | None = Field(
        default=None, description=". Maximum execution time in seconds"
    )
    respect_context_window: bool = Field(
        default=True,
        description="Whether to respect the context window of the LLM",
    )
    use_stop_words: bool = Field(
        default=True,
        description="Whether to use stop words to prevent the LLM from using tools",
    )
    request_within_rpm_limit: Callable[[], bool] | None = Field(
        default=None,
        description="Callback to check if the request is within the RPM8 limit",
    )
    i18n: I18N = Field(
        default_factory=get_i18n, description="Internationalization settings."
    )
    response_format: type[BaseModel] | None = Field(
        default=None, description="Pydantic model for structured output"
    )
    verbose: bool = Field(
        default=False, description="Whether to print execution details"
    )
    guardrail: GuardrailType | None = Field(
        default=None,
        description="Function or string description of a guardrail to validate agent output",
    )
    guardrail_max_retries: int = Field(
        default=3, description="Maximum number of retries when guardrail fails"
    )
    a2a: (
        list[A2AConfig | A2AServerConfig | A2AClientConfig]
        | A2AConfig
        | A2AServerConfig
        | A2AClientConfig
        | None
    ) = Field(
        default=None,
        description="A2A (Agent-to-Agent) configuration for delegating tasks to remote agents. "
        "Can be a single A2AConfig/A2AClientConfig/A2AServerConfig, or a list of configurations.",
    )
    tools_results: list[dict[str, Any]] = Field(
        default_factory=list, description="Results of the tools used by the agent."
    )
    original_agent: BaseAgent | None = Field(
        default=None, description="Reference to the agent that created this LiteAgent"
    )
    _parsed_tools: list[CrewStructuredTool] = PrivateAttr(default_factory=list)
    _token_process: TokenProcess = PrivateAttr(default_factory=TokenProcess)
    _cache_handler: CacheHandler = PrivateAttr(default_factory=CacheHandler)
    _key: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))
    _messages: list[LLMMessage] = PrivateAttr(default_factory=list)
    _iterations: int = PrivateAttr(default=0)
    _printer: Printer = PrivateAttr(default_factory=Printer)
    _guardrail: GuardrailCallable | None = PrivateAttr(default=None)
    _guardrail_retry_count: int = PrivateAttr(default=0)
    _callbacks: list[TokenCalcHandler] = PrivateAttr(default_factory=list)
    _before_llm_call_hooks: list[BeforeLLMCallHookType] = PrivateAttr(
        default_factory=get_before_llm_call_hooks
    )
    _after_llm_call_hooks: list[AfterLLMCallHookType] = PrivateAttr(
        default_factory=get_after_llm_call_hooks
    )

    @model_validator(mode="after")
    def emit_deprecation_warning(self) -> Self:
        """Emit deprecation warning for LiteAgent usage."""
        warnings.warn(
            "LiteAgent is deprecated and will be removed in a future version. "
            "Use Agent().kickoff(messages) instead, which provides the same "
            "functionality with additional features like memory and knowledge support.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self

    @model_validator(mode="after")
    def setup_llm(self) -> Self:
        """Set up the LLM and other components after initialization."""
        self.llm = create_llm(self.llm)
        if not isinstance(self.llm, BaseLLM):
            raise ValueError(
                f"Expected LLM instance of type BaseLLM, got {type(self.llm).__name__}"
            )
        token_callback = TokenCalcHandler(token_cost_process=self._token_process)
        self._callbacks = [token_callback]

        return self

    @model_validator(mode="after")
    def parse_tools(self) -> Self:
        """Parse the tools and convert them to CrewStructuredTool instances."""
        self._parsed_tools = parse_tools(self.tools)

        return self

    @model_validator(mode="after")
    def setup_a2a_support(self) -> Self:
        """Setup A2A extensions and server methods if a2a config exists."""
        if self.a2a:
            from crewai.a2a.config import A2AClientConfig, A2AConfig
            from crewai.a2a.extensions.registry import (
                create_extension_registry_from_config,
            )
            from crewai.a2a.utils.agent_card import inject_a2a_server_methods

            configs = self.a2a if isinstance(self.a2a, list) else [self.a2a]
            client_configs = [
                config
                for config in configs
                if isinstance(config, (A2AConfig, A2AClientConfig))
            ]

            extension_registry = (
                create_extension_registry_from_config(client_configs)
                if client_configs
                else create_extension_registry_from_config([])
            )
            extension_registry.inject_all_tools(self)  # type: ignore[arg-type]
            inject_a2a_server_methods(self)  # type: ignore[arg-type]

            original_kickoff = self.kickoff

            @wraps(original_kickoff)
            def kickoff_with_a2a(
                messages: str | list[LLMMessage],
                response_format: type[BaseModel] | None = None,
                input_files: dict[str, FileInput] | None = None,
            ) -> LiteAgentOutput:
                return _kickoff_with_a2a_support(
                    self,
                    original_kickoff,
                    messages,
                    response_format,
                    input_files,
                    extension_registry,
                )

            object.__setattr__(self, "kickoff", MethodType(kickoff_with_a2a, self))

        return self

    @model_validator(mode="after")
    def ensure_guardrail_is_callable(self) -> Self:
        if callable(self.guardrail):
            self._guardrail = self.guardrail
        elif isinstance(self.guardrail, str):
            from crewai.tasks.llm_guardrail import LLMGuardrail

            if not isinstance(self.llm, BaseLLM):
                raise TypeError(
                    f"Guardrail requires LLM instance of type BaseLLM, got {type(self.llm).__name__}"
                )
            self._guardrail = cast(
                GuardrailCallable,
                cast(object, LLMGuardrail(description=self.guardrail, llm=self.llm)),
            )

        return self

    @field_validator("guardrail", mode="before")
    @classmethod
    def validate_guardrail_function(
        cls, v: GuardrailCallable | str | None
    ) -> GuardrailCallable | str | None:
        """Validate that the guardrail function has the correct signature.

        If v is a callable, validate that it has the correct signature.
        If v is a string, return it as is.

        Args:
            v: The guardrail function to validate or a string describing the guardrail task

        Returns:
            The validated guardrail function or a string describing the guardrail task
        """
        if v is None or isinstance(v, str):
            return v

        # Check function signature
        sig = inspect.signature(v)
        if len(sig.parameters) != 1:
            raise ValueError(
                f"Guardrail function must accept exactly 1 parameter (LiteAgentOutput), "
                f"but it accepts {len(sig.parameters)}"
            )

        # Check return annotation if present
        if sig.return_annotation is not sig.empty:
            if sig.return_annotation == tuple[bool, Any]:
                return v

            origin = get_origin(sig.return_annotation)
            args = get_args(sig.return_annotation)

            if origin is not tuple or len(args) != 2 or args[0] is not bool:
                raise ValueError(
                    "If return type is annotated, it must be Tuple[bool, Any]"
                )

        return v

    @property
    def key(self) -> str:
        """Get the unique key for this agent instance."""
        return self._key

    @property
    def _original_role(self) -> str:
        """Return the original role for compatibility with tool interfaces."""
        return self.role

    @property
    def before_llm_call_hooks(self) -> list[BeforeLLMCallHookType]:
        """Get the before_llm_call hooks for this agent."""
        return self._before_llm_call_hooks

    @property
    def after_llm_call_hooks(self) -> list[AfterLLMCallHookType]:
        """Get the after_llm_call hooks for this agent."""
        return self._after_llm_call_hooks

    @property
    def messages(self) -> list[LLMMessage]:
        """Get the messages list for hook context compatibility."""
        return self._messages

    @property
    def iterations(self) -> int:
        """Get the current iteration count for hook context compatibility."""
        return self._iterations

    def kickoff(
        self,
        messages: str | list[LLMMessage],
        response_format: type[BaseModel] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> LiteAgentOutput:
        """Execute the agent with the given messages.

        Args:
            messages: Either a string query or a list of message dictionaries.
                     If a string is provided, it will be converted to a user message.
                     If a list is provided, each dict should have 'role' and 'content' keys.
            response_format: Optional Pydantic model for structured output. If provided,
                           overrides self.response_format for this execution.
            input_files: Optional dict of named files to attach to the message.
                   Files can be paths, bytes, or File objects from crewai_files.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        # Create agent info for event emission
        agent_info = {
            "id": self.id,
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
            self._messages = self._format_messages(
                messages, response_format=response_format, input_files=input_files
            )

            return self._execute_core(
                agent_info=agent_info, response_format=response_format
            )

        except Exception as e:
            if self.verbose:
                self._printer.print(
                    content="Agent failed to reach a final answer. This is likely a bug - please report it.",
                    color="red",
                )
            handle_unknown_error(self._printer, e, verbose=self.verbose)
            # Emit error event
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionErrorEvent(
                    agent_info=agent_info,
                    error=str(e),
                ),
            )
            raise e

    def _execute_core(
        self, agent_info: dict[str, Any], response_format: type[BaseModel] | None = None
    ) -> LiteAgentOutput:
        # Emit event for agent execution start
        crewai_event_bus.emit(
            self,
            event=LiteAgentExecutionStartedEvent(
                agent_info=agent_info,
                tools=self._parsed_tools,
                messages=self._messages,
            ),
        )

        # Execute the agent using invoke loop
        agent_finish = self._invoke_loop()
        formatted_result: BaseModel | None = None

        active_response_format = response_format or self.response_format
        if active_response_format:
            try:
                model_schema = generate_model_description(active_response_format)
                schema = json.dumps(model_schema, indent=2)
                instructions = self.i18n.slice("formatted_task_instructions").format(
                    output_format=schema
                )

                converter = Converter(
                    llm=self.llm,
                    text=agent_finish.output,
                    model=active_response_format,
                    instructions=instructions,
                )

                result = converter.to_pydantic()
                if isinstance(result, BaseModel):
                    formatted_result = result
            except ConverterError as e:
                if self.verbose:
                    self._printer.print(
                        content=f"Failed to parse output into response format after retries: {e.message}",
                        color="yellow",
                    )

        # Calculate token usage metrics
        if isinstance(self.llm, BaseLLM):
            usage_metrics = self.llm.get_token_usage_summary()
        else:
            usage_metrics = self._token_process.get_summary()

        # Create output
        output = LiteAgentOutput(
            raw=agent_finish.output,
            pydantic=formatted_result,
            agent_role=self.role,
            usage_metrics=usage_metrics.model_dump() if usage_metrics else None,
            messages=self._messages,
        )

        # Process guardrail if set
        if self._guardrail is not None:
            guardrail_result = process_guardrail(
                output=output,
                guardrail=self._guardrail,
                retry_count=self._guardrail_retry_count,
                event_source=self,
                from_agent=self,
            )

            if not guardrail_result.success:
                if self._guardrail_retry_count >= self.guardrail_max_retries:
                    raise Exception(
                        f"Agent's guardrail failed validation after {self.guardrail_max_retries} retries. "
                        f"Last error: {guardrail_result.error}"
                    )
                self._guardrail_retry_count += 1
                if self.verbose:
                    self._printer.print(
                        f"Guardrail failed. Retrying ({self._guardrail_retry_count}/{self.guardrail_max_retries})..."
                        f"\n{guardrail_result.error}"
                    )

                self._messages.append(
                    {
                        "role": "user",
                        "content": guardrail_result.error
                        or "Guardrail validation failed",
                    }
                )

                return self._execute_core(agent_info=agent_info)

            # Apply guardrail result if available
            if guardrail_result.result is not None:
                if isinstance(guardrail_result.result, str):
                    output.raw = guardrail_result.result
                elif isinstance(guardrail_result.result, BaseModel):
                    output.pydantic = guardrail_result.result

            if isinstance(self.llm, BaseLLM):
                usage_metrics = self.llm.get_token_usage_summary()
            else:
                usage_metrics = self._token_process.get_summary()
            output.usage_metrics = usage_metrics.model_dump() if usage_metrics else None

        # Emit completion event
        crewai_event_bus.emit(
            self,
            event=LiteAgentExecutionCompletedEvent(
                agent_info=agent_info,
                output=agent_finish.output,
            ),
        )

        return output

    async def kickoff_async(
        self,
        messages: str | list[LLMMessage],
        response_format: type[BaseModel] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> LiteAgentOutput:
        """Execute the agent asynchronously with the given messages.

        Args:
            messages: Either a string query or a list of message dictionaries.
                     If a string is provided, it will be converted to a user message.
                     If a list is provided, each dict should have 'role' and 'content' keys.
            response_format: Optional Pydantic model for structured output.
            input_files: Optional dict of named files to attach to the message.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        return await asyncio.to_thread(
            self.kickoff, messages, response_format, input_files
        )

    async def akickoff(
        self,
        messages: str | list[LLMMessage],
        response_format: type[BaseModel] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> LiteAgentOutput:
        """Async version of kickoff. Alias for kickoff_async.

        Args:
            messages: Either a string query or a list of message dictionaries.
            response_format: Optional Pydantic model for structured output.
            input_files: Optional dict of named files to attach to the message.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        return await self.kickoff_async(messages, response_format, input_files)

    def _get_default_system_prompt(
        self, response_format: type[BaseModel] | None = None
    ) -> str:
        """Get the default system prompt for the agent.

        Args:
            response_format: Optional response format to use instead of self.response_format
        """
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

        active_response_format = response_format or self.response_format
        if active_response_format:
            model_description = generate_model_description(active_response_format)
            schema_json = json.dumps(model_description, indent=2)
            base_prompt += self.i18n.slice("lite_agent_response_format").format(
                response_format=schema_json
            )

        return base_prompt

    def _format_messages(
        self,
        messages: str | list[LLMMessage],
        response_format: type[BaseModel] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> list[LLMMessage]:
        """Format messages for the LLM.

        Args:
            messages: Input messages to format.
            response_format: Optional response format to use instead of self.response_format.
            input_files: Optional dict of named files to include with the messages.
        """
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        system_prompt = self._get_default_system_prompt(response_format=response_format)

        # Add system message at the beginning
        formatted_messages: list[LLMMessage] = [
            {"role": "system", "content": system_prompt}
        ]

        # Add the rest of the messages
        formatted_messages.extend(messages)

        # Attach files to the last user message if provided
        if input_files:
            for msg in reversed(formatted_messages):
                if msg.get("role") == "user":
                    msg["files"] = input_files
                    break

        return formatted_messages

    def _invoke_loop(self) -> AgentFinish:
        """
        Run the agent's thought process until it reaches a conclusion or max iterations.

        Returns:
            AgentFinish: The final result of the agent execution.
        """
        # Execute the agent loop
        formatted_answer: AgentAction | AgentFinish | None = None
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
                        verbose=self.verbose,
                    )

                enforce_rpm_limit(self.request_within_rpm_limit)

                try:
                    answer = get_llm_response(
                        llm=cast(LLM, self.llm),
                        messages=self._messages,
                        callbacks=self._callbacks,
                        printer=self._printer,
                        from_agent=self,
                        executor_context=self,
                        verbose=self.verbose,
                    )

                except Exception as e:
                    raise e

                formatted_answer = process_llm_response(
                    cast(str, answer), self.use_stop_words
                )

                if isinstance(formatted_answer, AgentAction):
                    try:
                        tool_result = execute_tool_and_check_finality(
                            agent_action=formatted_answer,
                            tools=self._parsed_tools,
                            i18n=self.i18n,
                            agent_key=self.key,
                            agent_role=self.role,
                            agent=self.original_agent,
                            crew=None,
                        )
                    except Exception as e:
                        raise e

                    formatted_answer = handle_agent_action_core(
                        formatted_answer=formatted_answer,
                        tool_result=tool_result,
                        show_logs=self._show_logs,
                    )

                self._append_message(formatted_answer.text, role="assistant")
            except OutputParserError as e:  # noqa: PERF203
                if self.verbose:
                    self._printer.print(
                        content="Failed to parse LLM output. Retrying...",
                        color="yellow",
                    )
                formatted_answer = handle_output_parser_exception(
                    e=e,
                    messages=self._messages,
                    iterations=self._iterations,
                    log_error_after=3,
                    printer=self._printer,
                    verbose=self.verbose,
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
                        verbose=self.verbose,
                    )
                    continue
                handle_unknown_error(self._printer, e, verbose=self.verbose)
                raise e

            finally:
                self._iterations += 1

        if not isinstance(formatted_answer, AgentFinish):
            raise RuntimeError(
                "Agent execution ended without reaching a final answer. "
                f"Got {type(formatted_answer).__name__} instead of AgentFinish."
            )
        self._show_logs(formatted_answer)
        return formatted_answer

    def _show_logs(self, formatted_answer: AgentAction | AgentFinish) -> None:
        """Show logs for the agent's execution."""
        crewai_event_bus.emit(
            self,
            AgentLogsExecutionEvent(
                agent_role=self.role,
                formatted_answer=formatted_answer,
                verbose=self.verbose,
            ),
        )

    def _append_message(
        self, text: str, role: Literal["user", "assistant", "system"] = "assistant"
    ) -> None:
        """Append a message to the message list with the given role."""
        self._messages.append(format_message_for_llm(text, role=role))


try:
    from crewai.a2a.config import (
        A2AClientConfig as _A2AClientConfig,
        A2AConfig as _A2AConfig,
        A2AServerConfig as _A2AServerConfig,
    )

    LiteAgent.model_rebuild(
        _types_namespace={
            "A2AConfig": _A2AConfig,
            "A2AClientConfig": _A2AClientConfig,
            "A2AServerConfig": _A2AServerConfig,
        }
    )
except ImportError:
    pass
