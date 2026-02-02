from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
import shutil
import subprocess
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    cast,
)
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self

from crewai.agent.utils import (
    ahandle_knowledge_retrieval,
    apply_training_data,
    build_task_prompt_with_schema,
    format_task_with_context,
    get_knowledge_config,
    handle_knowledge_retrieval,
    handle_reasoning,
    prepare_tools,
    process_tool_results,
    save_last_messages,
    validate_max_execution_time,
)
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    LiteAgentExecutionCompletedEvent,
    LiteAgentExecutionErrorEvent,
    LiteAgentExecutionStartedEvent,
)
from crewai.events.types.knowledge_events import (
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
)
from crewai.events.types.memory_events import (
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalFailedEvent,
    MemoryRetrievalStartedEvent,
)
from crewai.experimental.agent_executor import AgentExecutor
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.lite_agent_output import LiteAgentOutput
from crewai.llms.base_llm import BaseLLM
from crewai.mcp import (
    MCPClient,
    MCPServerConfig,
    MCPServerHTTP,
    MCPServerSSE,
    MCPServerStdio,
)
from crewai.mcp.transports.http import HTTPTransport
from crewai.mcp.transports.sse import SSETransport
from crewai.mcp.transports.stdio import StdioTransport
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.rag.embeddings.types import EmbedderConfig
from crewai.security.fingerprint import Fingerprint
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities.agent_utils import (
    get_tool_names,
    is_inside_event_loop,
    load_agent_from_repository,
    parse_tools,
    render_text_description_and_args,
)
from crewai.utilities.constants import TRAINED_AGENTS_DATA_FILE, TRAINING_DATA_FILE
from crewai.utilities.converter import Converter, ConverterError
from crewai.utilities.guardrail import process_guardrail
from crewai.utilities.guardrail_types import GuardrailType
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.prompts import Prompts, StandardPromptResult, SystemPromptResult
from crewai.utilities.pydantic_schema_utils import generate_model_description
from crewai.utilities.string_utils import sanitize_tool_name
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.training_handler import CrewTrainingHandler


try:
    from crewai.a2a.types import AgentResponseProtocol
except ImportError:
    AgentResponseProtocol = None  # type: ignore[assignment, misc]


if TYPE_CHECKING:
    from crewai_files import FileInput
    from crewai_tools import CodeInterpreterTool

    from crewai.a2a.config import A2AClientConfig, A2AConfig, A2AServerConfig
    from crewai.agents.agent_builder.base_agent import PlatformAppOrAction
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool
    from crewai.utilities.types import LLMMessage


# MCP Connection timeout constants (in seconds)
MCP_CONNECTION_TIMEOUT: Final[int] = 10
MCP_TOOL_EXECUTION_TIMEOUT: Final[int] = 30
MCP_DISCOVERY_TIMEOUT: Final[int] = 15
MCP_MAX_RETRIES: Final[int] = 3

# Simple in-memory cache for MCP tool schemas (duration: 5 minutes)
_mcp_schema_cache: dict[str, Any] = {}
_cache_ttl: Final[int] = 300  # 5 minutes


class Agent(BaseAgent):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the CrewAgentExecutor or AgentExecutor class.
            role: The role of the agent.
            goal: The objective of the agent.
            backstory: The backstory of the agent.
            knowledge: The knowledge base of the agent.
            config: Dict representation of agent configuration.
            llm: The language model that will run the agent.
            function_calling_llm: The language model that will handle the tool calling for this agent, it overrides the crew function_calling_llm.
            max_iter: Maximum number of iterations for an agent to execute a task.
            max_rpm: Maximum number of requests per minute for the agent execution to be respected.
            verbose: Whether the agent execution should be in verbose mode.
            allow_delegation: Whether the agent is allowed to delegate tasks to other agents.
            tools: Tools at agents disposal
            step_callback: Callback to be executed after each step of the agent execution.
            knowledge_sources: Knowledge sources for the agent.
            embedder: Embedder configuration for the agent.
            apps: List of applications that the agent can access through CrewAI Platform.
            mcps: List of MCP server references for tool integration.
    """

    model_config = ConfigDict()

    _times_executed: int = PrivateAttr(default=0)
    _mcp_clients: list[Any] = PrivateAttr(default_factory=list)
    _last_messages: list[LLMMessage] = PrivateAttr(default_factory=list)
    max_execution_time: int | None = Field(
        default=None,
        description="Maximum execution time for an agent to execute a task",
    )
    step_callback: Any | None = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    use_system_prompt: bool | None = Field(
        default=True,
        description="Use system prompt for the agent.",
    )
    llm: str | InstanceOf[BaseLLM] | Any = Field(
        description="Language model that will run the agent.", default=None
    )
    function_calling_llm: str | InstanceOf[BaseLLM] | Any | None = Field(
        description="Language model that will run the agent.", default=None
    )
    system_template: str | None = Field(
        default=None, description="System format for the agent."
    )
    prompt_template: str | None = Field(
        default=None, description="Prompt format for the agent."
    )
    response_template: str | None = Field(
        default=None, description="Response format for the agent."
    )
    allow_code_execution: bool | None = Field(
        default=False, description="Enable code execution for the agent."
    )
    respect_context_window: bool = Field(
        default=True,
        description="Keep messages under the context window size by summarizing content.",
    )
    max_retry_limit: int = Field(
        default=2,
        description="Maximum number of retries for an agent to execute a task when an error occurs.",
    )
    multimodal: bool = Field(
        default=False,
        deprecated=True,
        description="[DEPRECATED, will be removed in v2.0 - pass files natively.] Whether the agent is multimodal.",
    )
    inject_date: bool = Field(
        default=False,
        description="Whether to automatically inject the current date into tasks.",
    )
    date_format: str = Field(
        default="%Y-%m-%d",
        description="Format string for date when inject_date is enabled.",
    )
    code_execution_mode: Literal["safe", "unsafe"] = Field(
        default="safe",
        description="Mode for code execution: 'safe' (using Docker) or 'unsafe' (direct execution).",
    )
    reasoning: bool = Field(
        default=False,
        description="Whether the agent should reflect and create a plan before executing a task.",
    )
    max_reasoning_attempts: int | None = Field(
        default=None,
        description="Maximum number of reasoning attempts before executing the task. If None, will try until ready.",
    )
    embedder: EmbedderConfig | None = Field(
        default=None,
        description="Embedder configuration for the agent.",
    )
    agent_knowledge_context: str | None = Field(
        default=None,
        description="Knowledge context for the agent.",
    )
    crew_knowledge_context: str | None = Field(
        default=None,
        description="Knowledge context for the crew.",
    )
    knowledge_search_query: str | None = Field(
        default=None,
        description="Knowledge search query for the agent dynamically generated by the agent.",
    )
    from_repository: str | None = Field(
        default=None,
        description="The Agent's role to be used from your repository.",
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
        description="""
        A2A (Agent-to-Agent) configuration for delegating tasks to remote agents.
        Can be a single A2AConfig/A2AClientConfig/A2AServerConfig, or a list of any number of A2AConfig/A2AClientConfig with a single A2AServerConfig.
        """,
    )
    executor_class: type[CrewAgentExecutor] | type[AgentExecutor] = Field(
        default=CrewAgentExecutor,
        description="Class to use for the agent executor. Defaults to CrewAgentExecutor, can optionally use AgentExecutor.",
    )

    @model_validator(mode="before")
    def validate_from_repository(cls, v: Any) -> dict[str, Any] | None | Any:  # noqa: N805
        if v is not None and (from_repository := v.get("from_repository")):
            return load_agent_from_repository(from_repository) | v
        return v

    @model_validator(mode="after")
    def post_init_setup(self) -> Self:
        self.llm = create_llm(self.llm)
        if self.function_calling_llm and not isinstance(
            self.function_calling_llm, BaseLLM
        ):
            self.function_calling_llm = create_llm(self.function_calling_llm)

        if not self.agent_executor:
            self._setup_agent_executor()

        if self.allow_code_execution:
            self._validate_docker_installation()

        return self

    def _setup_agent_executor(self) -> None:
        if not self.cache_handler:
            self.cache_handler = CacheHandler()
        self.set_cache_handler(self.cache_handler)

    def set_knowledge(self, crew_embedder: EmbedderConfig | None = None) -> None:
        try:
            if self.embedder is None and crew_embedder:
                self.embedder = crew_embedder

            if self.knowledge_sources:
                if isinstance(self.knowledge_sources, list) and all(
                    isinstance(k, BaseKnowledgeSource) for k in self.knowledge_sources
                ):
                    self.knowledge = Knowledge(
                        sources=self.knowledge_sources,
                        embedder=self.embedder,
                        collection_name=self.role,
                    )
                    self.knowledge.add_sources()
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid Knowledge Configuration: {e!s}") from e

    def _is_any_available_memory(self) -> bool:
        """Check if any memory is available."""
        if not self.crew:
            return False

        memory_attributes = [
            "memory",
            "_short_term_memory",
            "_long_term_memory",
            "_entity_memory",
            "_external_memory",
        ]

        return any(getattr(self.crew, attr) for attr in memory_attributes)

    def _supports_native_tool_calling(self, tools: list[BaseTool]) -> bool:
        """Check if the LLM supports native function calling with the given tools.

        Args:
            tools: List of tools to check against.

        Returns:
            True if native function calling is supported and tools are available.
        """
        return (
            hasattr(self.llm, "supports_function_calling")
            and callable(getattr(self.llm, "supports_function_calling", None))
            and self.llm.supports_function_calling()
            and len(tools) > 0
        )

    def execute_task(
        self,
        task: Task,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> Any:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent

        Raises:
            TimeoutError: If execution exceeds the maximum execution time.
            ValueError: If the max execution time is not a positive integer.
            RuntimeError: If the agent execution fails for other reasons.
        """
        handle_reasoning(self, task)
        self._inject_date_to_task(task)

        if self.tools_handler:
            self.tools_handler.last_used_tool = None

        task_prompt = task.prompt()
        task_prompt = build_task_prompt_with_schema(task, task_prompt, self.i18n)
        task_prompt = format_task_with_context(task_prompt, context, self.i18n)

        if self._is_any_available_memory():
            crewai_event_bus.emit(
                self,
                event=MemoryRetrievalStartedEvent(
                    task_id=str(task.id) if task else None,
                    source_type="agent",
                    from_agent=self,
                    from_task=task,
                ),
            )

            start_time = time.time()
            memory = ""

            try:
                contextual_memory = ContextualMemory(
                    self.crew._short_term_memory,
                    self.crew._long_term_memory,
                    self.crew._entity_memory,
                    self.crew._external_memory,
                    agent=self,
                    task=task,
                )
                memory = contextual_memory.build_context_for_task(task, context or "")
                if memory.strip() != "":
                    task_prompt += self.i18n.slice("memory").format(memory=memory)

                crewai_event_bus.emit(
                    self,
                    event=MemoryRetrievalCompletedEvent(
                        task_id=str(task.id) if task else None,
                        memory_content=memory,
                        retrieval_time_ms=(time.time() - start_time) * 1000,
                        source_type="agent",
                        from_agent=self,
                        from_task=task,
                    ),
                )
            except Exception as e:
                crewai_event_bus.emit(
                    self,
                    event=MemoryRetrievalFailedEvent(
                        task_id=str(task.id) if task else None,
                        source_type="agent",
                        from_agent=self,
                        from_task=task,
                        error=str(e),
                    ),
                )

        knowledge_config = get_knowledge_config(self)
        task_prompt = handle_knowledge_retrieval(
            self,
            task,
            task_prompt,
            knowledge_config,
            self.knowledge.query if self.knowledge else lambda *a, **k: None,
            self.crew.query_knowledge if self.crew else lambda *a, **k: None,
        )

        prepare_tools(self, tools, task)
        task_prompt = apply_training_data(self, task_prompt)

        from crewai.events.types.agent_events import (
            AgentExecutionCompletedEvent,
            AgentExecutionErrorEvent,
            AgentExecutionStartedEvent,
        )

        try:
            crewai_event_bus.emit(
                self,
                event=AgentExecutionStartedEvent(
                    agent=self,
                    tools=self.tools,
                    task_prompt=task_prompt,
                    task=task,
                ),
            )

            validate_max_execution_time(self.max_execution_time)
            if self.max_execution_time is not None:
                result = self._execute_with_timeout(
                    task_prompt, task, self.max_execution_time
                )
            else:
                result = self._execute_without_timeout(task_prompt, task)

        except TimeoutError as e:
            crewai_event_bus.emit(
                self,
                event=AgentExecutionErrorEvent(
                    agent=self,
                    task=task,
                    error=str(e),
                ),
            )
            raise e
        except Exception as e:
            if e.__class__.__module__.startswith("litellm"):
                crewai_event_bus.emit(
                    self,
                    event=AgentExecutionErrorEvent(
                        agent=self,
                        task=task,
                        error=str(e),
                    ),
                )
                raise e
            self._times_executed += 1
            if self._times_executed > self.max_retry_limit:
                crewai_event_bus.emit(
                    self,
                    event=AgentExecutionErrorEvent(
                        agent=self,
                        task=task,
                        error=str(e),
                    ),
                )
                raise e
            result = self.execute_task(task, context, tools)

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        result = process_tool_results(self, result)

        output_for_event = result
        if (
            AgentResponseProtocol is not None
            and isinstance(result, BaseModel)
            and isinstance(result, AgentResponseProtocol)
        ):
            output_for_event = str(result.message)
        elif not isinstance(result, str):
            output_for_event = str(result)

        crewai_event_bus.emit(
            self,
            event=AgentExecutionCompletedEvent(
                agent=self, task=task, output=output_for_event
            ),
        )

        save_last_messages(self)
        self._cleanup_mcp_clients()

        return result

    def _execute_with_timeout(self, task_prompt: str, task: Task, timeout: int) -> Any:
        """Execute a task with a timeout.

        Args:
            task_prompt: The prompt to send to the agent.
            task: The task being executed.
            timeout: Maximum execution time in seconds.

        Returns:
            The output of the agent.

        Raises:
            TimeoutError: If execution exceeds the timeout.
            RuntimeError: If execution fails for other reasons.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                self._execute_without_timeout, task_prompt=task_prompt, task=task
            )

            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as e:
                future.cancel()
                raise TimeoutError(
                    f"Task '{task.description}' execution timed out after {timeout} seconds. Consider increasing max_execution_time or optimizing the task."
                ) from e
            except Exception as e:
                future.cancel()
                raise RuntimeError(f"Task execution failed: {e!s}") from e

    def _execute_without_timeout(self, task_prompt: str, task: Task) -> Any:
        """Execute a task without a timeout.

        Args:
            task_prompt: The prompt to send to the agent.
            task: The task being executed.

        Returns:
            The output of the agent.
        """
        if not self.agent_executor:
            raise RuntimeError("Agent executor is not initialized.")

        return self.agent_executor.invoke(
            {
                "input": task_prompt,
                "tool_names": self.agent_executor.tools_names,
                "tools": self.agent_executor.tools_description,
                "ask_for_human_input": task.human_input,
            }
        )["output"]

    async def aexecute_task(
        self,
        task: Task,
        context: str | None = None,
        tools: list[BaseTool] | None = None,
    ) -> Any:
        """Execute a task with the agent asynchronously.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent.

        Raises:
            TimeoutError: If execution exceeds the maximum execution time.
            ValueError: If the max execution time is not a positive integer.
            RuntimeError: If the agent execution fails for other reasons.
        """
        handle_reasoning(self, task)
        self._inject_date_to_task(task)

        if self.tools_handler:
            self.tools_handler.last_used_tool = None

        task_prompt = task.prompt()
        task_prompt = build_task_prompt_with_schema(task, task_prompt, self.i18n)
        task_prompt = format_task_with_context(task_prompt, context, self.i18n)

        if self._is_any_available_memory():
            crewai_event_bus.emit(
                self,
                event=MemoryRetrievalStartedEvent(
                    task_id=str(task.id) if task else None,
                    source_type="agent",
                    from_agent=self,
                    from_task=task,
                ),
            )

            start_time = time.time()
            memory = ""

            try:
                contextual_memory = ContextualMemory(
                    self.crew._short_term_memory,
                    self.crew._long_term_memory,
                    self.crew._entity_memory,
                    self.crew._external_memory,
                    agent=self,
                    task=task,
                )
                memory = await contextual_memory.abuild_context_for_task(
                    task, context or ""
                )
                if memory.strip() != "":
                    task_prompt += self.i18n.slice("memory").format(memory=memory)

                crewai_event_bus.emit(
                    self,
                    event=MemoryRetrievalCompletedEvent(
                        task_id=str(task.id) if task else None,
                        memory_content=memory,
                        retrieval_time_ms=(time.time() - start_time) * 1000,
                        source_type="agent",
                        from_agent=self,
                        from_task=task,
                    ),
                )
            except Exception as e:
                crewai_event_bus.emit(
                    self,
                    event=MemoryRetrievalFailedEvent(
                        task_id=str(task.id) if task else None,
                        source_type="agent",
                        from_agent=self,
                        from_task=task,
                        error=str(e),
                    ),
                )

        knowledge_config = get_knowledge_config(self)
        task_prompt = await ahandle_knowledge_retrieval(
            self, task, task_prompt, knowledge_config
        )

        prepare_tools(self, tools, task)
        task_prompt = apply_training_data(self, task_prompt)

        from crewai.events.types.agent_events import (
            AgentExecutionCompletedEvent,
            AgentExecutionErrorEvent,
            AgentExecutionStartedEvent,
        )

        try:
            crewai_event_bus.emit(
                self,
                event=AgentExecutionStartedEvent(
                    agent=self,
                    tools=self.tools,
                    task_prompt=task_prompt,
                    task=task,
                ),
            )

            validate_max_execution_time(self.max_execution_time)
            if self.max_execution_time is not None:
                result = await self._aexecute_with_timeout(
                    task_prompt, task, self.max_execution_time
                )
            else:
                result = await self._aexecute_without_timeout(task_prompt, task)

        except TimeoutError as e:
            crewai_event_bus.emit(
                self,
                event=AgentExecutionErrorEvent(
                    agent=self,
                    task=task,
                    error=str(e),
                ),
            )
            raise e
        except Exception as e:
            if e.__class__.__module__.startswith("litellm"):
                crewai_event_bus.emit(
                    self,
                    event=AgentExecutionErrorEvent(
                        agent=self,
                        task=task,
                        error=str(e),
                    ),
                )
                raise e
            self._times_executed += 1
            if self._times_executed > self.max_retry_limit:
                crewai_event_bus.emit(
                    self,
                    event=AgentExecutionErrorEvent(
                        agent=self,
                        task=task,
                        error=str(e),
                    ),
                )
                raise e
            result = await self.aexecute_task(task, context, tools)

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        result = process_tool_results(self, result)

        output_for_event = result
        if (
            AgentResponseProtocol is not None
            and isinstance(result, BaseModel)
            and isinstance(result, AgentResponseProtocol)
        ):
            output_for_event = str(result.message)
        elif not isinstance(result, str):
            output_for_event = str(result)

        crewai_event_bus.emit(
            self,
            event=AgentExecutionCompletedEvent(
                agent=self, task=task, output=output_for_event
            ),
        )

        save_last_messages(self)
        self._cleanup_mcp_clients()

        return result

    async def _aexecute_with_timeout(
        self, task_prompt: str, task: Task, timeout: int
    ) -> Any:
        """Execute a task with a timeout asynchronously.

        Args:
            task_prompt: The prompt to send to the agent.
            task: The task being executed.
            timeout: Maximum execution time in seconds.

        Returns:
            The output of the agent.

        Raises:
            TimeoutError: If execution exceeds the timeout.
            RuntimeError: If execution fails for other reasons.
        """
        try:
            return await asyncio.wait_for(
                self._aexecute_without_timeout(task_prompt, task),
                timeout=timeout,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"Task '{task.description}' execution timed out after {timeout} seconds. "
                "Consider increasing max_execution_time or optimizing the task."
            ) from e

    async def _aexecute_without_timeout(self, task_prompt: str, task: Task) -> Any:
        """Execute a task without a timeout asynchronously.

        Args:
            task_prompt: The prompt to send to the agent.
            task: The task being executed.

        Returns:
            The output of the agent.
        """
        if not self.agent_executor:
            raise RuntimeError("Agent executor is not initialized.")

        result = await self.agent_executor.ainvoke(
            {
                "input": task_prompt,
                "tool_names": self.agent_executor.tools_names,
                "tools": self.agent_executor.tools_description,
                "ask_for_human_input": task.human_input,
            }
        )
        return result["output"]

    def create_agent_executor(
        self, tools: list[BaseTool] | None = None, task: Task | None = None
    ) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        raw_tools: list[BaseTool] = tools or self.tools or []
        parsed_tools = parse_tools(raw_tools)

        use_native_tool_calling = self._supports_native_tool_calling(raw_tools)

        prompt = Prompts(
            agent=self,
            has_tools=len(raw_tools) > 0,
            use_native_tool_calling=use_native_tool_calling,
            i18n=self.i18n,
            use_system_prompt=self.use_system_prompt,
            system_template=self.system_template,
            prompt_template=self.prompt_template,
            response_template=self.response_template,
        ).task_execution()

        stop_words = [self.i18n.slice("observation")]

        if self.response_template:
            stop_words.append(
                self.response_template.split("{{ .Response }}")[1].strip()
            )

        rpm_limit_fn = (
            self._rpm_controller.check_or_wait if self._rpm_controller else None
        )

        if self.agent_executor is not None:
            self._update_executor_parameters(
                task=task,
                tools=parsed_tools,  # type: ignore[arg-type]
                raw_tools=raw_tools,
                prompt=prompt,
                stop_words=stop_words,
                rpm_limit_fn=rpm_limit_fn,
            )
        else:
            self.agent_executor = self.executor_class(
                llm=cast(BaseLLM, self.llm),
                task=task,  # type: ignore[arg-type]
                i18n=self.i18n,
                agent=self,
                crew=self.crew,
                tools=parsed_tools,
                prompt=prompt,
                original_tools=raw_tools,
                stop_words=stop_words,
                max_iter=self.max_iter,
                tools_handler=self.tools_handler,
                tools_names=get_tool_names(parsed_tools),
                tools_description=render_text_description_and_args(parsed_tools),
                step_callback=self.step_callback,
                function_calling_llm=self.function_calling_llm,
                respect_context_window=self.respect_context_window,
                request_within_rpm_limit=rpm_limit_fn,
                callbacks=[TokenCalcHandler(self._token_process)],
                response_model=task.response_model if task else None,
            )

    def _update_executor_parameters(
        self,
        task: Task | None,
        tools: list[BaseTool],
        raw_tools: list[BaseTool],
        prompt: SystemPromptResult | StandardPromptResult,
        stop_words: list[str],
        rpm_limit_fn: Callable | None,  # type: ignore[type-arg]
    ) -> None:
        """Update executor parameters without recreating instance.

        Args:
            task: Task to execute.
            tools: Parsed tools.
            raw_tools: Original tools.
            prompt: Generated prompt.
            stop_words: Stop words list.
            rpm_limit_fn: RPM limit callback function.
        """
        self.agent_executor.task = task
        self.agent_executor.tools = tools
        self.agent_executor.original_tools = raw_tools
        self.agent_executor.prompt = prompt
        self.agent_executor.stop = stop_words
        self.agent_executor.tools_names = get_tool_names(tools)
        self.agent_executor.tools_description = render_text_description_and_args(tools)
        self.agent_executor.response_model = task.response_model if task else None

        self.agent_executor.tools_handler = self.tools_handler
        self.agent_executor.request_within_rpm_limit = rpm_limit_fn

        if self.agent_executor.llm:
            existing_stop = getattr(self.agent_executor.llm, "stop", [])
            self.agent_executor.llm.stop = list(
                set(
                    existing_stop + stop_words
                    if isinstance(existing_stop, list)
                    else stop_words
                )
            )

    def get_delegation_tools(self, agents: list[BaseAgent]) -> list[BaseTool]:
        agent_tools = AgentTools(agents=agents)
        return agent_tools.tools()

    def get_platform_tools(self, apps: list[PlatformAppOrAction]) -> list[BaseTool]:
        try:
            from crewai_tools import (
                CrewaiPlatformTools,
            )

            return CrewaiPlatformTools(apps=apps)
        except Exception as e:
            self._logger.log("error", f"Error getting platform tools: {e!s}")
            return []

    def get_mcp_tools(self, mcps: list[str | MCPServerConfig]) -> list[BaseTool]:
        """Convert MCP server references/configs to CrewAI tools.

        Supports both string references (backwards compatible) and structured
        configuration objects (MCPServerStdio, MCPServerHTTP, MCPServerSSE).

        Args:
            mcps: List of MCP server references (strings) or configurations.

        Returns:
            List of BaseTool instances from MCP servers.
        """
        all_tools = []
        clients = []

        for mcp_config in mcps:
            if isinstance(mcp_config, str):
                tools = self._get_mcp_tools_from_string(mcp_config)
            else:
                tools, client = self._get_native_mcp_tools(mcp_config)
                if client:
                    clients.append(client)

            all_tools.extend(tools)

        # Store clients for cleanup
        self._mcp_clients.extend(clients)
        return all_tools

    def _cleanup_mcp_clients(self) -> None:
        """Cleanup MCP client connections after task execution."""
        if not self._mcp_clients:
            return

        async def _disconnect_all() -> None:
            for client in self._mcp_clients:
                if client and hasattr(client, "connected") and client.connected:
                    await client.disconnect()

        try:
            asyncio.run(_disconnect_all())
        except Exception as e:
            self._logger.log("error", f"Error during MCP client cleanup: {e}")
        finally:
            self._mcp_clients.clear()

    def _get_mcp_tools_from_string(self, mcp_ref: str) -> list[BaseTool]:
        """Get tools from legacy string-based MCP references.

        This method maintains backwards compatibility with string-based
        MCP references (https://... and crewai-amp:...).

        Args:
            mcp_ref: String reference to MCP server.

        Returns:
            List of BaseTool instances.
        """
        if mcp_ref.startswith("crewai-amp:"):
            return self._get_amp_mcp_tools(mcp_ref)
        if mcp_ref.startswith("https://"):
            return self._get_external_mcp_tools(mcp_ref)
        return []

    def _get_external_mcp_tools(self, mcp_ref: str) -> list[BaseTool]:
        """Get tools from external HTTPS MCP server with graceful error handling."""
        from crewai.tools.mcp_tool_wrapper import MCPToolWrapper

        # Parse server URL and optional tool name
        if "#" in mcp_ref:
            server_url, specific_tool = mcp_ref.split("#", 1)
        else:
            server_url, specific_tool = mcp_ref, None

        server_params = {"url": server_url}
        server_name = self._extract_server_name(server_url)

        try:
            # Get tool schemas with timeout and error handling
            tool_schemas = self._get_mcp_tool_schemas(server_params)

            if not tool_schemas:
                self._logger.log(
                    "warning", f"No tools discovered from MCP server: {server_url}"
                )
                return []

            tools = []
            for tool_name, schema in tool_schemas.items():
                # Skip if specific tool requested and this isn't it
                if specific_tool and tool_name != specific_tool:
                    continue

                try:
                    wrapper = MCPToolWrapper(
                        mcp_server_params=server_params,
                        tool_name=tool_name,
                        tool_schema=schema,
                        server_name=server_name,
                    )
                    tools.append(wrapper)
                except Exception as e:
                    self._logger.log(
                        "warning",
                        f"Failed to create MCP tool wrapper for {tool_name}: {e}",
                    )
                    continue

            if specific_tool and not tools:
                self._logger.log(
                    "warning",
                    f"Specific tool '{specific_tool}' not found on MCP server: {server_url}",
                )

            return cast(list[BaseTool], tools)

        except Exception as e:
            self._logger.log(
                "warning", f"Failed to connect to MCP server {server_url}: {e}"
            )
            return []

    def _get_native_mcp_tools(
        self, mcp_config: MCPServerConfig
    ) -> tuple[list[BaseTool], Any | None]:
        """Get tools from MCP server using structured configuration.

        This method creates an MCP client based on the configuration type,
        connects to the server, discovers tools, applies filtering, and
        returns wrapped tools along with the client instance for cleanup.

        Args:
            mcp_config: MCP server configuration (MCPServerStdio, MCPServerHTTP, or MCPServerSSE).

        Returns:
            Tuple of (list of BaseTool instances, MCPClient instance for cleanup).
        """
        from crewai.tools.base_tool import BaseTool
        from crewai.tools.mcp_native_tool import MCPNativeTool

        transport: StdioTransport | HTTPTransport | SSETransport
        if isinstance(mcp_config, MCPServerStdio):
            transport = StdioTransport(
                command=mcp_config.command,
                args=mcp_config.args,
                env=mcp_config.env,
            )
            server_name = f"{mcp_config.command}_{'_'.join(mcp_config.args)}"
        elif isinstance(mcp_config, MCPServerHTTP):
            transport = HTTPTransport(
                url=mcp_config.url,
                headers=mcp_config.headers,
                streamable=mcp_config.streamable,
            )
            server_name = self._extract_server_name(mcp_config.url)
        elif isinstance(mcp_config, MCPServerSSE):
            transport = SSETransport(
                url=mcp_config.url,
                headers=mcp_config.headers,
            )
            server_name = self._extract_server_name(mcp_config.url)
        else:
            raise ValueError(f"Unsupported MCP server config type: {type(mcp_config)}")

        client = MCPClient(
            transport=transport,
            cache_tools_list=mcp_config.cache_tools_list,
        )

        async def _setup_client_and_list_tools() -> list[dict[str, Any]]:
            """Async helper to connect and list tools in same event loop."""

            try:
                if not client.connected:
                    await client.connect()

                tools_list = await client.list_tools()

                try:
                    await client.disconnect()
                    # Small delay to allow background tasks to finish cleanup
                    # This helps prevent "cancel scope in different task" errors
                    # when asyncio.run() closes the event loop
                    await asyncio.sleep(0.1)
                except Exception as e:
                    self._logger.log("error", f"Error during disconnect: {e}")

                return tools_list
            except Exception as e:
                if client.connected:
                    await client.disconnect()
                    await asyncio.sleep(0.1)
                raise RuntimeError(
                    f"Error during setup client and list tools: {e}"
                ) from e

        try:
            try:
                asyncio.get_running_loop()
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, _setup_client_and_list_tools()
                    )
                    tools_list = future.result()
            except RuntimeError:
                try:
                    tools_list = asyncio.run(_setup_client_and_list_tools())
                except RuntimeError as e:
                    error_msg = str(e).lower()
                    if "cancel scope" in error_msg or "task" in error_msg:
                        raise ConnectionError(
                            "MCP connection failed due to event loop cleanup issues. "
                            "This may be due to authentication errors or server unavailability."
                        ) from e
                except asyncio.CancelledError as e:
                    raise ConnectionError(
                        "MCP connection was cancelled. This may indicate an authentication "
                        "error or server unavailability."
                    ) from e

            if mcp_config.tool_filter:
                filtered_tools = []
                for tool in tools_list:
                    if callable(mcp_config.tool_filter):
                        try:
                            from crewai.mcp.filters import ToolFilterContext

                            context = ToolFilterContext(
                                agent=self,
                                server_name=server_name,
                                run_context=None,
                            )
                            if mcp_config.tool_filter(context, tool):  # type: ignore[call-arg, arg-type]
                                filtered_tools.append(tool)
                        except (TypeError, AttributeError):
                            if mcp_config.tool_filter(tool):  # type: ignore[call-arg, arg-type]
                                filtered_tools.append(tool)
                    else:
                        # Not callable - include tool
                        filtered_tools.append(tool)
                tools_list = filtered_tools

            tools = []
            for tool_def in tools_list:
                tool_name = tool_def.get("name", "")
                if not tool_name:
                    continue

                # Convert inputSchema to Pydantic model if present
                args_schema = None
                if tool_def.get("inputSchema"):
                    args_schema = self._json_schema_to_pydantic(
                        tool_name, tool_def["inputSchema"]
                    )

                tool_schema = {
                    "description": tool_def.get("description", ""),
                    "args_schema": args_schema,
                }

                try:
                    native_tool = MCPNativeTool(
                        mcp_client=client,
                        tool_name=tool_name,
                        tool_schema=tool_schema,
                        server_name=server_name,
                    )
                    tools.append(native_tool)
                except Exception as e:
                    self._logger.log("error", f"Failed to create native MCP tool: {e}")
                    continue

            return cast(list[BaseTool], tools), client
        except Exception as e:
            if client.connected:
                asyncio.run(client.disconnect())

            raise RuntimeError(f"Failed to get native MCP tools: {e}") from e

    def _get_amp_mcp_tools(self, amp_ref: str) -> list[BaseTool]:
        """Get tools from CrewAI AMP MCP marketplace."""
        # Parse: "crewai-amp:mcp-name" or "crewai-amp:mcp-name#tool_name"
        amp_part = amp_ref.replace("crewai-amp:", "")
        if "#" in amp_part:
            mcp_name, specific_tool = amp_part.split("#", 1)
        else:
            mcp_name, specific_tool = amp_part, None

        # Call AMP API to get MCP server URLs
        mcp_servers = self._fetch_amp_mcp_servers(mcp_name)

        tools = []
        for server_config in mcp_servers:
            server_ref = server_config["url"]
            if specific_tool:
                server_ref += f"#{specific_tool}"
            server_tools = self._get_external_mcp_tools(server_ref)
            tools.extend(server_tools)

        return tools

    @staticmethod
    def _extract_server_name(server_url: str) -> str:
        """Extract clean server name from URL for tool prefixing."""

        parsed = urlparse(server_url)
        domain = parsed.netloc.replace(".", "_")
        path = parsed.path.replace("/", "_").strip("_")
        return f"{domain}_{path}" if path else domain

    def _get_mcp_tool_schemas(
        self, server_params: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Get tool schemas from MCP server for wrapper creation with caching."""
        server_url = server_params["url"]

        # Check cache first
        cache_key = server_url
        current_time = time.time()

        if cache_key in _mcp_schema_cache:
            cached_data, cache_time = _mcp_schema_cache[cache_key]
            if current_time - cache_time < _cache_ttl:
                self._logger.log(
                    "debug", f"Using cached MCP tool schemas for {server_url}"
                )
                return cached_data  # type: ignore[no-any-return]

        try:
            schemas = asyncio.run(self._get_mcp_tool_schemas_async(server_params))

            # Cache successful results
            _mcp_schema_cache[cache_key] = (schemas, current_time)

            return schemas
        except Exception as e:
            # Log warning but don't raise - this allows graceful degradation
            self._logger.log(
                "warning", f"Failed to get MCP tool schemas from {server_url}: {e}"
            )
            return {}

    async def _get_mcp_tool_schemas_async(
        self, server_params: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Async implementation of MCP tool schema retrieval with timeouts and retries."""
        server_url = server_params["url"]
        return await self._retry_mcp_discovery(
            self._discover_mcp_tools_with_timeout, server_url
        )

    async def _retry_mcp_discovery(
        self, operation_func: Any, server_url: str
    ) -> dict[str, dict[str, Any]]:
        """Retry MCP discovery operation with exponential backoff, avoiding try-except in loop."""
        last_error = None

        for attempt in range(MCP_MAX_RETRIES):
            # Execute single attempt outside try-except loop structure
            result, error, should_retry = await self._attempt_mcp_discovery(
                operation_func, server_url
            )

            # Success case - return immediately
            if result is not None:
                return result

            # Non-retryable error - raise immediately
            if not should_retry:
                raise RuntimeError(error)

            # Retryable error - continue with backoff
            last_error = error
            if attempt < MCP_MAX_RETRIES - 1:
                wait_time = 2**attempt  # Exponential backoff
                await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"Failed to discover MCP tools after {MCP_MAX_RETRIES} attempts: {last_error}"
        )

    @staticmethod
    async def _attempt_mcp_discovery(
        operation_func: Any, server_url: str
    ) -> tuple[dict[str, dict[str, Any]] | None, str, bool]:
        """Attempt single MCP discovery operation and return (result, error_message, should_retry)."""
        try:
            result = await operation_func(server_url)
            return result, "", False

        except ImportError:
            return (
                None,
                "MCP library not available. Please install with: pip install mcp",
                False,
            )

        except asyncio.TimeoutError:
            return (
                None,
                f"MCP discovery timed out after {MCP_DISCOVERY_TIMEOUT} seconds",
                True,
            )

        except Exception as e:
            error_str = str(e).lower()

            # Classify errors as retryable or non-retryable
            if "authentication" in error_str or "unauthorized" in error_str:
                return None, f"Authentication failed for MCP server: {e!s}", False
            if "connection" in error_str or "network" in error_str:
                return None, f"Network connection failed: {e!s}", True
            if "json" in error_str or "parsing" in error_str:
                return None, f"Server response parsing error: {e!s}", True
            return None, f"MCP discovery error: {e!s}", False

    async def _discover_mcp_tools_with_timeout(
        self, server_url: str
    ) -> dict[str, dict[str, Any]]:
        """Discover MCP tools with timeout wrapper."""
        return await asyncio.wait_for(
            self._discover_mcp_tools(server_url), timeout=MCP_DISCOVERY_TIMEOUT
        )

    async def _discover_mcp_tools(self, server_url: str) -> dict[str, dict[str, Any]]:
        """Discover tools from MCP server with proper timeout handling."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(server_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                # Initialize the connection with timeout
                await asyncio.wait_for(
                    session.initialize(), timeout=MCP_CONNECTION_TIMEOUT
                )

                # List available tools with timeout
                tools_result = await asyncio.wait_for(
                    session.list_tools(),
                    timeout=MCP_DISCOVERY_TIMEOUT - MCP_CONNECTION_TIMEOUT,
                )

                schemas = {}
                for tool in tools_result.tools:
                    args_schema = None
                    if hasattr(tool, "inputSchema") and tool.inputSchema:
                        args_schema = self._json_schema_to_pydantic(
                            sanitize_tool_name(tool.name), tool.inputSchema
                        )

                    schemas[sanitize_tool_name(tool.name)] = {
                        "description": getattr(tool, "description", ""),
                        "args_schema": args_schema,
                    }
                return schemas

    def _json_schema_to_pydantic(
        self, tool_name: str, json_schema: dict[str, Any]
    ) -> type:
        """Convert JSON Schema to Pydantic model for tool arguments.

        Args:
            tool_name: Name of the tool (used for model naming)
            json_schema: JSON Schema dict with 'properties', 'required', etc.

        Returns:
            Pydantic BaseModel class
        """
        from pydantic import Field, create_model

        properties = json_schema.get("properties", {})
        required_fields = json_schema.get("required", [])

        field_definitions: dict[str, Any] = {}

        for field_name, field_schema in properties.items():
            field_type = self._json_type_to_python(field_schema)
            field_description = field_schema.get("description", "")

            is_required = field_name in required_fields

            if is_required:
                field_definitions[field_name] = (
                    field_type,
                    Field(..., description=field_description),
                )
            else:
                field_definitions[field_name] = (
                    field_type | None,
                    Field(default=None, description=field_description),
                )

        model_name = f"{tool_name.replace('-', '_').replace(' ', '_')}Schema"
        return create_model(model_name, **field_definitions)  # type: ignore[no-any-return]

    def _json_type_to_python(self, field_schema: dict[str, Any]) -> type:
        """Convert JSON Schema type to Python type.

        Args:
            field_schema: JSON Schema field definition

        Returns:
            Python type
        """

        json_type = field_schema.get("type")

        if "anyOf" in field_schema:
            types: list[type] = []
            for option in field_schema["anyOf"]:
                if "const" in option:
                    types.append(str)
                else:
                    types.append(self._json_type_to_python(option))
            unique_types = list(set(types))
            if len(unique_types) > 1:
                result: Any = unique_types[0]
                for t in unique_types[1:]:
                    result = result | t
                return result  # type: ignore[no-any-return]
            return unique_types[0]

        type_mapping: dict[str | None, type] = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        return type_mapping.get(json_type, Any)

    @staticmethod
    def _fetch_amp_mcp_servers(mcp_name: str) -> list[dict[str, Any]]:
        """Fetch MCP server configurations from CrewAI AMP API."""
        # TODO: Implement AMP API call to "integrations/mcps" endpoint
        # Should return list of server configs with URLs
        return []

    @staticmethod
    def get_multimodal_tools() -> Sequence[BaseTool]:
        from crewai.tools.agent_tools.add_image_tool import AddImageTool

        return [AddImageTool()]

    def get_code_execution_tools(self) -> list[CodeInterpreterTool]:
        try:
            from crewai_tools import (
                CodeInterpreterTool,
            )

            # Set the unsafe_mode based on the code_execution_mode attribute
            unsafe_mode = self.code_execution_mode == "unsafe"
            return [CodeInterpreterTool(unsafe_mode=unsafe_mode)]
        except ModuleNotFoundError:
            self._logger.log(
                "info", "Coding tools not available. Install crewai_tools. "
            )
            return []

    @staticmethod
    def get_output_converter(
        llm: BaseLLM, text: str, model: type[BaseModel], instructions: str
    ) -> Converter:
        return Converter(llm=llm, text=text, model=model, instructions=instructions)

    def _training_handler(self, task_prompt: str) -> str:
        """Handle training data for the agent task prompt to improve output on Training."""
        if data := CrewTrainingHandler(TRAINING_DATA_FILE).load():
            agent_id = str(self.id)

            if data.get(agent_id):
                human_feedbacks = [
                    i["human_feedback"] for i in data.get(agent_id, {}).values()
                ]
                task_prompt += (
                    "\n\nYou MUST follow these instructions: \n "
                    + "\n - ".join(human_feedbacks)
                )

        return task_prompt

    def _use_trained_data(self, task_prompt: str) -> str:
        """Use trained data for the agent task prompt to improve output."""
        if data := CrewTrainingHandler(TRAINED_AGENTS_DATA_FILE).load():
            if trained_data_output := data.get(self.role):
                task_prompt += (
                    "\n\nYou MUST follow these instructions: \n - "
                    + "\n - ".join(trained_data_output["suggestions"])
                )
        return task_prompt

    @staticmethod
    def _render_text_description(tools: list[Any]) -> str:
        """Render the tool name and description in plain text.

        Output will be in the format of:

        .. code-block:: markdown

            search: This tool is used for search
            calculator: This tool is used for math
        """
        return "\n".join(
            [
                f"Tool name: {sanitize_tool_name(tool.name)}\nTool description:\n{tool.description}"
                for tool in tools
            ]
        )

    def _inject_date_to_task(self, task: Task) -> None:
        """Inject the current date into the task description if inject_date is enabled."""
        if self.inject_date:
            from datetime import datetime

            try:
                valid_format_codes = [
                    "%Y",
                    "%m",
                    "%d",
                    "%H",
                    "%M",
                    "%S",
                    "%B",
                    "%b",
                    "%A",
                    "%a",
                ]
                is_valid = any(code in self.date_format for code in valid_format_codes)

                if not is_valid:
                    raise ValueError(f"Invalid date format: {self.date_format}")

                current_date = datetime.now().strftime(self.date_format)
                task.description += f"\n\nCurrent Date: {current_date}"
            except Exception as e:
                self._logger.log("warning", f"Failed to inject date: {e!s}")

    def _validate_docker_installation(self) -> None:
        """Check if Docker is installed and running."""
        docker_path = shutil.which("docker")
        if not docker_path:
            raise RuntimeError(
                f"Docker is not installed. Please install Docker to use code execution with agent: {self.role}"
            )

        try:
            subprocess.run(  # noqa: S603
                [docker_path, "info"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Docker is not running. Please start Docker to use code execution with agent: {self.role}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                f"Docker command timed out. Please check your Docker installation for agent: {self.role}"
            ) from e

    def __repr__(self) -> str:
        return f"Agent(role={self.role}, goal={self.goal}, backstory={self.backstory})"

    @property
    def fingerprint(self) -> Fingerprint:
        """
        Get the agent's fingerprint.

        Returns:
            Fingerprint: The agent's fingerprint
        """
        return self.security_config.fingerprint

    def set_fingerprint(self, fingerprint: Fingerprint) -> None:
        self.security_config.fingerprint = fingerprint

    @property
    def last_messages(self) -> list[LLMMessage]:
        """Get messages from the last task execution.

        Returns:
            List of LLM messages from the most recent task execution.
        """
        return self._last_messages

    def _get_knowledge_search_query(self, task_prompt: str, task: Task) -> str | None:
        """Generate a search query for the knowledge base based on the task description."""
        crewai_event_bus.emit(
            self,
            event=KnowledgeQueryStartedEvent(
                task_prompt=task_prompt,
                from_task=task,
                from_agent=self,
            ),
        )
        query = self.i18n.slice("knowledge_search_query").format(
            task_prompt=task_prompt
        )
        rewriter_prompt = self.i18n.slice("knowledge_search_query_system_prompt")
        if not isinstance(self.llm, BaseLLM):
            self._logger.log(
                "warning",
                f"Knowledge search query failed: LLM for agent '{self.role}' is not an instance of BaseLLM",
            )
            crewai_event_bus.emit(
                self,
                event=KnowledgeQueryFailedEvent(
                    error="LLM is not compatible with knowledge search queries",
                    from_task=task,
                    from_agent=self,
                ),
            )
            return None

        try:
            rewritten_query = self.llm.call(
                [
                    {
                        "role": "system",
                        "content": rewriter_prompt,
                    },
                    {"role": "user", "content": query},
                ]
            )
            crewai_event_bus.emit(
                self,
                event=KnowledgeQueryCompletedEvent(
                    query=query,
                    from_task=task,
                    from_agent=self,
                ),
            )
            return rewritten_query
        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=KnowledgeQueryFailedEvent(
                    error=str(e),
                    from_task=task,
                    from_agent=self,
                ),
            )
            return None

    def _prepare_kickoff(
        self,
        messages: str | list[LLMMessage],
        response_format: type[Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> tuple[AgentExecutor, dict[str, Any], dict[str, Any], list[CrewStructuredTool]]:
        """Prepare common setup for kickoff execution.

        This method handles all the common preparation logic shared between
        kickoff() and kickoff_async(), including tool processing, prompt building,
        executor creation, and input formatting.

        Args:
            messages: Either a string query or a list of message dictionaries.
            response_format: Optional Pydantic model for structured output.
            input_files: Optional dict of named files to attach to the message.

        Returns:
            Tuple of (executor, inputs, agent_info, parsed_tools) ready for execution.
        """
        # Process platform apps and MCP tools
        if self.apps:
            platform_tools = self.get_platform_tools(self.apps)
            if platform_tools and self.tools is not None:
                self.tools.extend(platform_tools)
        if self.mcps:
            mcps = self.get_mcp_tools(self.mcps)
            if mcps and self.tools is not None:
                self.tools.extend(mcps)

        # Prepare tools
        raw_tools: list[BaseTool] = self.tools or []
        parsed_tools = parse_tools(raw_tools)

        # Build agent_info for backward-compatible event emission
        agent_info = {
            "id": self.id,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": raw_tools,
            "verbose": self.verbose,
        }

        # Build prompt for standalone execution
        use_native_tool_calling = self._supports_native_tool_calling(raw_tools)
        prompt = Prompts(
            agent=self,
            has_tools=len(raw_tools) > 0,
            use_native_tool_calling=use_native_tool_calling,
            i18n=self.i18n,
            use_system_prompt=self.use_system_prompt,
            system_template=self.system_template,
            prompt_template=self.prompt_template,
            response_template=self.response_template,
        ).task_execution()

        # Prepare stop words
        stop_words = [self.i18n.slice("observation")]
        if self.response_template:
            stop_words.append(
                self.response_template.split("{{ .Response }}")[1].strip()
            )

        # Get RPM limit function
        rpm_limit_fn = (
            self._rpm_controller.check_or_wait if self._rpm_controller else None
        )

        # Create the executor for standalone mode (no crew, no task)
        executor = AgentExecutor(
            task=None,
            crew=None,
            llm=cast(BaseLLM, self.llm),
            agent=self,
            prompt=prompt,
            max_iter=self.max_iter,
            tools=parsed_tools,
            tools_names=get_tool_names(parsed_tools),
            stop_words=stop_words,
            tools_description=render_text_description_and_args(parsed_tools),
            tools_handler=self.tools_handler,
            original_tools=raw_tools,
            step_callback=self.step_callback,
            function_calling_llm=self.function_calling_llm,
            respect_context_window=self.respect_context_window,
            request_within_rpm_limit=rpm_limit_fn,
            callbacks=[TokenCalcHandler(self._token_process)],
            response_model=response_format,
            i18n=self.i18n,
        )

        all_files: dict[str, Any] = {}
        if isinstance(messages, str):
            formatted_messages = messages
        else:
            formatted_messages = "\n".join(
                str(msg.get("content", "")) for msg in messages if msg.get("content")
            )
            for msg in messages:
                if msg.get("files"):
                    all_files.update(msg["files"])

        if input_files:
            all_files.update(input_files)

        # Build the input dict for the executor
        inputs: dict[str, Any] = {
            "input": formatted_messages,
            "tool_names": get_tool_names(parsed_tools),
            "tools": render_text_description_and_args(parsed_tools),
        }
        if all_files:
            inputs["files"] = all_files

        return executor, inputs, agent_info, parsed_tools

    def kickoff(
        self,
        messages: str | list[LLMMessage],
        response_format: type[Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> LiteAgentOutput | Coroutine[Any, Any, LiteAgentOutput]:
        """Execute the agent with the given messages using the AgentExecutor.

        This method provides standalone agent execution without requiring a Crew.
        It supports tools, response formatting, guardrails, and file inputs.

        When called from within a Flow (sync or async method), this automatically
        detects the event loop and returns a coroutine that the Flow framework
        awaits. Users don't need to handle async explicitly.

        Args:
            messages: Either a string query or a list of message dictionaries.
                     If a string is provided, it will be converted to a user message.
                     If a list is provided, each dict should have 'role' and 'content' keys.
                     Messages can include a 'files' field with file inputs.
            response_format: Optional Pydantic model for structured output.
            input_files: Optional dict of named files to attach to the message.
                   Files can be paths, bytes, or File objects from crewai_files.

        Returns:
            LiteAgentOutput: The result of the agent execution.
            When inside a Flow, returns a coroutine that resolves to LiteAgentOutput.

        Note:
            For explicit async usage outside of Flow, use kickoff_async() directly.
        """
        # Magic auto-async: if inside event loop (e.g., inside a Flow),
        # return coroutine for Flow to await
        if is_inside_event_loop():
            return self.kickoff_async(messages, response_format, input_files)

        executor, inputs, agent_info, parsed_tools = self._prepare_kickoff(
            messages, response_format, input_files
        )

        try:
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionStartedEvent(
                    agent_info=agent_info,
                    tools=parsed_tools,
                    messages=messages,
                ),
            )

            output = self._execute_and_build_output(executor, inputs, response_format)
            if self.guardrail is not None:
                output = self._process_kickoff_guardrail(
                    output=output,
                    executor=executor,
                    inputs=inputs,
                    response_format=response_format,
                )

            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionCompletedEvent(
                    agent_info=agent_info,
                    output=output.raw,
                ),
            )

            return output

        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionErrorEvent(
                    agent_info=agent_info,
                    error=str(e),
                ),
            )
            raise

    def _execute_and_build_output(
        self,
        executor: AgentExecutor,
        inputs: dict[str, str],
        response_format: type[Any] | None = None,
    ) -> LiteAgentOutput:
        """Execute the agent and build the output object.

        Args:
            executor: The executor instance.
            inputs: Input dictionary for execution.
            response_format: Optional response format.

        Returns:
            LiteAgentOutput with raw output, formatted result, and metrics.
        """
        import json

        # Execute the agent (this is called from sync path, so invoke returns dict)
        result = cast(dict[str, Any], executor.invoke(inputs))
        output = result.get("output", "")

        # Handle response format conversion
        formatted_result: BaseModel | None = None
        raw_output: str

        if isinstance(output, BaseModel):
            formatted_result = output
            raw_output = output.model_dump_json()
        elif response_format:
            raw_output = str(output) if not isinstance(output, str) else output
            try:
                model_schema = generate_model_description(response_format)
                schema = json.dumps(model_schema, indent=2)
                instructions = self.i18n.slice("formatted_task_instructions").format(
                    output_format=schema
                )

                converter = Converter(
                    llm=self.llm,
                    text=raw_output,
                    model=response_format,
                    instructions=instructions,
                )

                conversion_result = converter.to_pydantic()
                if isinstance(conversion_result, BaseModel):
                    formatted_result = conversion_result
            except ConverterError:
                pass  # Keep raw output if conversion fails
        else:
            raw_output = str(output) if not isinstance(output, str) else output

        # Get token usage metrics
        if isinstance(self.llm, BaseLLM):
            usage_metrics = self.llm.get_token_usage_summary()
        else:
            usage_metrics = self._token_process.get_summary()

        raw_str = (
            raw_output
            if isinstance(raw_output, str)
            else raw_output.model_dump_json()
            if isinstance(raw_output, BaseModel)
            else str(raw_output)
        )

        return LiteAgentOutput(
            raw=raw_str,
            pydantic=formatted_result,
            agent_role=self.role,
            usage_metrics=usage_metrics.model_dump() if usage_metrics else None,
            messages=executor.messages,
        )

    async def _execute_and_build_output_async(
        self,
        executor: AgentExecutor,
        inputs: dict[str, str],
        response_format: type[Any] | None = None,
    ) -> LiteAgentOutput:
        """Execute the agent asynchronously and build the output object.

        This is the async version of _execute_and_build_output that uses
        invoke_async() for native async execution within event loops.

        Args:
            executor: The executor instance.
            inputs: Input dictionary for execution.
            response_format: Optional response format.

        Returns:
            LiteAgentOutput with raw output, formatted result, and metrics.
        """
        import json

        # Execute the agent asynchronously
        result = await executor.invoke_async(inputs)
        output = result.get("output", "")

        # Handle response format conversion
        formatted_result: BaseModel | None = None
        raw_output: str

        if isinstance(output, BaseModel):
            formatted_result = output
            raw_output = output.model_dump_json()
        elif response_format:
            raw_output = str(output) if not isinstance(output, str) else output
            try:
                model_schema = generate_model_description(response_format)
                schema = json.dumps(model_schema, indent=2)
                instructions = self.i18n.slice("formatted_task_instructions").format(
                    output_format=schema
                )

                converter = Converter(
                    llm=self.llm,
                    text=raw_output,
                    model=response_format,
                    instructions=instructions,
                )

                conversion_result = converter.to_pydantic()
                if isinstance(conversion_result, BaseModel):
                    formatted_result = conversion_result
            except ConverterError:
                pass  # Keep raw output if conversion fails
        else:
            raw_output = str(output) if not isinstance(output, str) else output

        # Get token usage metrics
        if isinstance(self.llm, BaseLLM):
            usage_metrics = self.llm.get_token_usage_summary()
        else:
            usage_metrics = self._token_process.get_summary()

        raw_str = (
            raw_output
            if isinstance(raw_output, str)
            else raw_output.model_dump_json()
            if isinstance(raw_output, BaseModel)
            else str(raw_output)
        )

        return LiteAgentOutput(
            raw=raw_str,
            pydantic=formatted_result,
            agent_role=self.role,
            usage_metrics=usage_metrics.model_dump() if usage_metrics else None,
            messages=executor.messages,
        )

    def _process_kickoff_guardrail(
        self,
        output: LiteAgentOutput,
        executor: AgentExecutor,
        inputs: dict[str, str],
        response_format: type[Any] | None = None,
        retry_count: int = 0,
    ) -> LiteAgentOutput:
        """Process guardrail for kickoff execution with retry logic.

        Args:
            output: Current agent output.
            executor: The executor instance.
            inputs: Input dictionary for re-execution.
            response_format: Optional response format.
            retry_count: Current retry count.

        Returns:
            Validated/updated output.
        """
        from crewai.utilities.guardrail_types import GuardrailCallable

        # Ensure guardrail is callable
        guardrail_callable: GuardrailCallable
        if isinstance(self.guardrail, str):
            from crewai.tasks.llm_guardrail import LLMGuardrail

            guardrail_callable = cast(
                GuardrailCallable,
                LLMGuardrail(description=self.guardrail, llm=cast(BaseLLM, self.llm)),
            )
        elif callable(self.guardrail):
            guardrail_callable = self.guardrail
        else:
            # Should not happen if called from kickoff with guardrail check
            return output

        guardrail_result = process_guardrail(
            output=output,
            guardrail=guardrail_callable,
            retry_count=retry_count,
            event_source=self,
            from_agent=self,
        )

        if not guardrail_result.success:
            if retry_count >= self.guardrail_max_retries:
                raise ValueError(
                    f"Agent's guardrail failed validation after {self.guardrail_max_retries} retries. "
                    f"Last error: {guardrail_result.error}"
                )

            # Add feedback and re-execute
            executor._append_message_to_state(
                guardrail_result.error or "Guardrail validation failed",
                role="user",
            )

            # Re-execute and build new output
            output = self._execute_and_build_output(executor, inputs, response_format)

            # Recursively retry guardrail
            return self._process_kickoff_guardrail(
                output=output,
                executor=executor,
                inputs=inputs,
                response_format=response_format,
                retry_count=retry_count + 1,
            )

        # Apply guardrail result if available
        if guardrail_result.result is not None:
            if isinstance(guardrail_result.result, str):
                output.raw = guardrail_result.result
            elif isinstance(guardrail_result.result, BaseModel):
                output.pydantic = guardrail_result.result

        return output

    async def kickoff_async(
        self,
        messages: str | list[LLMMessage],
        response_format: type[Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
    ) -> LiteAgentOutput:
        """Execute the agent asynchronously with the given messages.

        This is the async version of the kickoff method that uses native async
        execution. It is designed for use within async contexts, such as when
        called from within an async Flow method.

        Args:
            messages: Either a string query or a list of message dictionaries.
                     If a string is provided, it will be converted to a user message.
                     If a list is provided, each dict should have 'role' and 'content' keys.
                     Messages can include a 'files' field with file inputs.
            response_format: Optional Pydantic model for structured output.
            input_files: Optional dict of named files to attach to the message.
                   Files can be paths, bytes, or File objects from crewai_files.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        executor, inputs, agent_info, parsed_tools = self._prepare_kickoff(
            messages, response_format, input_files
        )

        try:
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionStartedEvent(
                    agent_info=agent_info,
                    tools=parsed_tools,
                    messages=messages,
                ),
            )

            output = await self._execute_and_build_output_async(
                executor, inputs, response_format
            )

            if self.guardrail is not None:
                output = self._process_kickoff_guardrail(
                    output=output,
                    executor=executor,
                    inputs=inputs,
                    response_format=response_format,
                )

            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionCompletedEvent(
                    agent_info=agent_info,
                    output=output.raw,
                ),
            )

            return output

        except Exception as e:
            crewai_event_bus.emit(
                self,
                event=LiteAgentExecutionErrorEvent(
                    agent_info=agent_info,
                    error=str(e),
                ),
            )
            raise

    async def akickoff(
        self,
        messages: str | list[LLMMessage],
        response_format: type[Any] | None = None,
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


# Rebuild Agent model to resolve A2A type forward references
try:
    from crewai.a2a.config import (
        A2AClientConfig as _A2AClientConfig,
        A2AConfig as _A2AConfig,
        A2AServerConfig as _A2AServerConfig,
    )

    Agent.model_rebuild(
        _types_namespace={
            "A2AConfig": _A2AConfig,
            "A2AClientConfig": _A2AClientConfig,
            "A2AServerConfig": _A2AServerConfig,
        }
    )
except ImportError:
    pass
