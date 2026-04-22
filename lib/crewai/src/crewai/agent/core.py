"""Core agent implementation for the CrewAI framework."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine, Sequence
import concurrent.futures
import contextvars
from datetime import datetime
import json
from pathlib import Path
import time
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NoReturn,
    cast,
)
import warnings

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    model_validator,
)
from pydantic.functional_serializers import PlainSerializer
from typing_extensions import Self, TypeIs

from crewai.agent.planning_config import PlanningConfig
from crewai.agent.utils import (
    ahandle_knowledge_retrieval,
    append_skill_context,
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
from crewai.agents.agent_builder.base_agent import (
    BaseAgent,
    _serialize_llm_ref,
    _validate_llm_ref,
)
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
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
from crewai.events.types.skill_events import SkillActivatedEvent
from crewai.experimental.agent_executor import AgentExecutor
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.lite_agent_output import LiteAgentOutput
from crewai.llms.base_llm import BaseLLM
from crewai.mcp.config import MCPServerConfig
from crewai.rag.embeddings.types import EmbedderConfig
from crewai.security.fingerprint import Fingerprint
from crewai.skills.loader import activate_skill, discover_skills
from crewai.skills.models import INSTRUCTIONS, Skill as SkillModel
from crewai.state.checkpoint_config import CheckpointConfig, apply_checkpoint
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.types.callback import SerializableCallable
from crewai.utilities.agent_utils import (
    get_tool_names,
    is_inside_event_loop,
    load_agent_from_repository,
    parse_tools,
    render_text_description_and_args,
)
from crewai.utilities.constants import TRAINED_AGENTS_DATA_FILE, TRAINING_DATA_FILE
from crewai.utilities.converter import Converter, ConverterError
from crewai.utilities.env import get_env_context
from crewai.utilities.guardrail import process_guardrail
from crewai.utilities.guardrail_types import GuardrailCallable, GuardrailType
from crewai.utilities.i18n import I18N_DEFAULT
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

    from crewai.a2a.config import A2AClientConfig, A2AConfig, A2AServerConfig
    from crewai.agents.agent_builder.base_agent import PlatformAppOrAction
    from crewai.mcp.tool_resolver import MCPToolResolver
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.tools.structured_tool import CrewStructuredTool
    from crewai.utilities.types import LLMMessage


_passthrough_exceptions: tuple[type[Exception], ...] = ()

_EXECUTOR_CLASS_MAP: dict[str, type] = {
    "CrewAgentExecutor": CrewAgentExecutor,
    "AgentExecutor": AgentExecutor,
}


def _is_resuming_agent_executor(
    executor: CrewAgentExecutor | AgentExecutor | None,
) -> TypeIs[AgentExecutor]:
    """Type guard: True when the executor is resuming from a checkpoint."""
    return isinstance(executor, AgentExecutor) and executor._resuming


def _validate_executor_class(value: Any) -> Any:
    if isinstance(value, str):
        cls = _EXECUTOR_CLASS_MAP.get(value)
        if cls is None:
            raise ValueError(f"Unknown executor class: {value}")
        return cls
    return value


def _serialize_executor_class(value: Any) -> str:
    return value.__name__ if isinstance(value, type) else str(value)


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
    _mcp_resolver: MCPToolResolver | None = PrivateAttr(default=None)
    _last_messages: list[LLMMessage] = PrivateAttr(default_factory=list)
    max_execution_time: int | None = Field(
        default=None,
        description="Maximum execution time for an agent to execute a task",
    )
    step_callback: SerializableCallable | None = Field(
        default=None,
        description="Callback to be executed after each step of the agent execution.",
    )
    use_system_prompt: bool | None = Field(
        default=True,
        description="Use system prompt for the agent.",
    )
    llm: Annotated[
        str | BaseLLM | None,
        BeforeValidator(_validate_llm_ref),
        PlainSerializer(_serialize_llm_ref, return_type=dict | None, when_used="json"),
    ] = Field(description="Language model that will run the agent.", default=None)
    function_calling_llm: Annotated[
        str | BaseLLM | None,
        BeforeValidator(_validate_llm_ref),
        PlainSerializer(_serialize_llm_ref, return_type=dict | None, when_used="json"),
    ] = Field(description="Language model that will run the agent.", default=None)
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
        default=False,
        deprecated=True,
        description="Deprecated. CodeInterpreterTool is no longer available. Use dedicated sandbox services instead.",
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
        deprecated=True,
        description="Deprecated. CodeInterpreterTool is no longer available. Use dedicated sandbox services instead.",
    )
    planning_config: PlanningConfig | None = Field(
        default=None,
        description="Configuration for agent planning before task execution.",
    )
    planning: bool = Field(
        default=False,
        description="Whether the agent should reflect and create a plan before executing a task.",
    )
    reasoning: bool = Field(
        default=False,
        description="[DEPRECATED: Use planning_config instead] Whether the agent should reflect and create a plan before executing a task.",
        deprecated=True,
    )
    max_reasoning_attempts: int | None = Field(
        default=None,
        description="[DEPRECATED: Use planning_config.max_attempts instead] Maximum number of reasoning attempts before executing the task. If None, will try until ready.",
        deprecated=True,
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
    agent_executor: CrewAgentExecutor | AgentExecutor | None = Field(
        default=None, description="An instance of the CrewAgentExecutor class."
    )
    executor_class: Annotated[
        type[CrewAgentExecutor] | type[AgentExecutor],
        BeforeValidator(_validate_executor_class),
        PlainSerializer(_serialize_executor_class, return_type=str, when_used="json"),
    ] = Field(
        default=CrewAgentExecutor,
        description="Class to use for the agent executor. Defaults to CrewAgentExecutor, can optionally use AgentExecutor.",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_from_repository(cls, v: Any) -> dict[str, Any] | None | Any:
        """Merge repository agent config with provided values before validation."""
        if v is not None and (from_repository := v.get("from_repository")):
            return load_agent_from_repository(from_repository) | v
        return v

    @model_validator(mode="after")
    def post_init_setup(self) -> Self:
        """Initialize LLM, executor, code tools, and skills after model creation."""
        self.llm = create_llm(self.llm)
        if self.function_calling_llm and not isinstance(
            self.function_calling_llm, BaseLLM
        ):
            self.function_calling_llm = create_llm(self.function_calling_llm)

        if not self.agent_executor:
            self._setup_agent_executor()

        if self.allow_code_execution:
            warnings.warn(
                "allow_code_execution is deprecated and will be removed in v2.0. "
                "CodeInterpreterTool is no longer available. "
                "Use dedicated sandbox services like E2B or Modal.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.set_skills()

        if self.reasoning and self.planning_config is None:
            warnings.warn(
                "The 'reasoning' parameter is deprecated. Use 'planning_config=PlanningConfig()' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.planning_config = PlanningConfig(
                max_attempts=self.max_reasoning_attempts,
            )

        return self

    @property
    def planning_enabled(self) -> bool:
        """Check if planning is enabled for this agent."""
        return self.planning_config is not None or self.planning

    def _setup_agent_executor(self) -> None:
        """Initialize the agent executor with a default cache handler."""
        if not self.cache_handler:
            self.cache_handler = CacheHandler()
        self.set_cache_handler(self.cache_handler)

    def set_knowledge(self, crew_embedder: EmbedderConfig | None = None) -> None:
        """Initialize knowledge sources with the agent or crew embedder config."""
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

    def set_skills(
        self,
        resolved_crew_skills: list[SkillModel] | None = None,
    ) -> None:
        """Resolve skill paths while preserving explicit disclosure levels.

        Path entries trigger discovery and activation because directory-based
        skills opt into eager loading. Pre-loaded Skill objects keep their
        current disclosure level so callers can attach METADATA-only skills and
        progressively activate them later. Crew-level skills are merged in with
        event emission so observability is consistent regardless of origin.

        Args:
            resolved_crew_skills: Pre-resolved crew skills. When provided,
                avoids redundant discovery per agent.
        """
        from crewai.crew import Crew

        if resolved_crew_skills is None:
            crew_skills: list[Path | SkillModel] | None = (
                self.crew.skills
                if isinstance(self.crew, Crew) and isinstance(self.crew.skills, list)
                else None
            )
        else:
            crew_skills = list(resolved_crew_skills)

        if not self.skills and not crew_skills:
            return

        needs_work = self.skills and any(
            isinstance(s, Path)
            or (isinstance(s, SkillModel) and s.disclosure_level < INSTRUCTIONS)
            for s in self.skills
        )
        if not needs_work and not crew_skills:
            return

        seen: set[str] = set()
        resolved: list[Path | SkillModel] = []
        items: list[Path | SkillModel] = list(self.skills) if self.skills else []

        if crew_skills:
            items.extend(crew_skills)

        for item in items:
            if isinstance(item, Path):
                discovered = discover_skills(item, source=self)
                for skill in discovered:
                    if skill.name not in seen:
                        seen.add(skill.name)
                        resolved.append(activate_skill(skill, source=self))
            elif isinstance(item, SkillModel):
                if item.name not in seen:
                    seen.add(item.name)
                    if item.disclosure_level >= INSTRUCTIONS:
                        crewai_event_bus.emit(
                            self,
                            event=SkillActivatedEvent(
                                from_agent=self,
                                skill_name=item.name,
                                skill_path=item.path,
                                disclosure_level=item.disclosure_level,
                            ),
                        )
                    resolved.append(item)

        self.skills = resolved if resolved else None

    def _is_any_available_memory(self) -> bool:
        """Check if unified memory is available (agent or crew)."""
        if getattr(self, "memory", None):
            return True
        if self.crew and getattr(self.crew, "_memory", None):
            return True
        return False

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
            and self.llm.supports_function_calling()  # type: ignore[union-attr]
            and len(tools) > 0
        )

    def _prepare_task_execution(
        self,
        task: Task,
        context: str | None,
    ) -> str:
        """Prepare common setup for task execution shared by sync and async paths.

        Handles reasoning, date injection, prompt building, and memory retrieval.

        Args:
            task: Task to execute.
            context: Context to execute the task in.

        Returns:
            The task prompt after memory retrieval, ready for knowledge lookup.
        """
        get_env_context()
        if self.executor_class is not AgentExecutor:
            handle_reasoning(self, task)

        self._inject_date_to_task(task)

        if self.tools_handler:
            self.tools_handler.last_used_tool = None

        task_prompt = task.prompt()
        task_prompt = build_task_prompt_with_schema(task, task_prompt)
        task_prompt = format_task_with_context(task_prompt, context)
        return self._retrieve_memory_context(task, task_prompt)

    def _finalize_task_prompt(
        self,
        task_prompt: str,
        tools: list[BaseTool] | None,
        task: Task,
    ) -> str:
        """Apply skill context, tool preparation, and training data to the task prompt.

        Args:
            task_prompt: The task prompt after memory and knowledge retrieval.
            tools: Tools to use for the task.
            task: Task to execute.

        Returns:
            The fully prepared task prompt.
        """
        task_prompt = append_skill_context(self, task_prompt)
        prepare_tools(self, tools, task)

        return apply_training_data(self, task_prompt)

    def _retrieve_memory_context(self, task: Task, task_prompt: str) -> str:
        """Retrieve memory context and append it to the task prompt.

        Args:
            task: The task being executed.
            task_prompt: The current task prompt.

        Returns:
            The task prompt, potentially augmented with memory context.
        """
        if not self._is_any_available_memory():
            return task_prompt

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
            unified_memory = getattr(self, "memory", None) or (
                getattr(self.crew, "_memory", None) if self.crew else None
            )
            if unified_memory is not None:
                query = task.description
                matches = unified_memory.recall(query, limit=5)
                if matches:
                    memory = "Relevant memories:\n" + "\n".join(
                        m.format() for m in matches
                    )
            if memory.strip() != "":
                task_prompt += I18N_DEFAULT.slice("memory").format(memory=memory)

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

        return task_prompt

    def _finalize_task_execution(self, task: Task, result: Any) -> Any:
        """Finalize task execution with RPM cleanup, tool processing, and event emission.

        Args:
            task: The task that was executed.
            result: The raw execution result.

        Returns:
            The processed result.
        """
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

    def _check_execution_error(self, e: Exception, task: Task) -> None:
        """Check if an execution error should be re-raised immediately.

        Args:
            e: The exception that occurred.
            task: The task being executed.

        Raises:
            Exception: If the error is from litellm, a passthrough, or retries are exhausted.
        """
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
        if isinstance(e, _passthrough_exceptions):
            raise
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

    def _handle_execution_error(
        self,
        e: Exception,
        task: Task,
        context: str | None,
        tools: list[BaseTool] | None,
    ) -> Any:
        """Handle execution errors with retry logic (sync path).

        Args:
            e: The exception that occurred.
            task: The task being executed.
            context: Task context.
            tools: Task tools.

        Returns:
            Result from retried execution.
        """
        self._check_execution_error(e, task)
        return self.execute_task(task, context, tools)

    async def _handle_execution_error_async(
        self,
        e: Exception,
        task: Task,
        context: str | None,
        tools: list[BaseTool] | None,
    ) -> Any:
        """Handle execution errors with retry logic (async path).

        Args:
            e: The exception that occurred.
            task: The task being executed.
            context: Task context.
            tools: Task tools.

        Returns:
            Result from retried execution.
        """
        self._check_execution_error(e, task)
        return await self.aexecute_task(task, context, tools)

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
        task_prompt = self._prepare_task_execution(task, context)

        knowledge_config = get_knowledge_config(self)
        task_prompt = handle_knowledge_retrieval(
            self,
            task,
            task_prompt,
            knowledge_config,
            self.knowledge.query if self.knowledge else lambda *a, **k: None,
            self.crew.query_knowledge
            if self.crew and not isinstance(self.crew, str)
            else lambda *a, **k: None,
        )

        task_prompt = self._finalize_task_prompt(task_prompt, tools, task)

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
            result = self._handle_execution_error(e, task, context, tools)

        return self._finalize_task_execution(task, result)

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
        ctx = contextvars.copy_context()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                ctx.run,
                self._execute_without_timeout,
                task_prompt=task_prompt,
                task=task,
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

        result = cast(
            dict[str, Any],
            self.agent_executor.invoke(
                {
                    "input": task_prompt,
                    "tool_names": self.agent_executor.tools_names,
                    "tools": self.agent_executor.tools_description,
                    "ask_for_human_input": task.human_input,
                }
            ),
        )
        return result["output"]

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
        task_prompt = self._prepare_task_execution(task, context)

        knowledge_config = get_knowledge_config(self)
        task_prompt = await ahandle_knowledge_retrieval(
            self, task, task_prompt, knowledge_config
        )

        task_prompt = self._finalize_task_prompt(task_prompt, tools, task)

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
            result = await self._handle_execution_error_async(e, task, context, tools)

        return self._finalize_task_execution(task, result)

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

    def _build_execution_prompt(
        self, raw_tools: list[BaseTool]
    ) -> tuple[
        SystemPromptResult | StandardPromptResult, list[str], Callable[[], bool] | None
    ]:
        """Build the execution prompt, stop words, and RPM limit function.

        Args:
            raw_tools: The raw tools available to the agent.

        Returns:
            A tuple of (prompt, stop_words, rpm_limit_fn).
        """
        use_native_tool_calling = self._supports_native_tool_calling(raw_tools)

        prompt = Prompts(
            agent=self,
            has_tools=len(raw_tools) > 0,
            use_native_tool_calling=use_native_tool_calling,
            use_system_prompt=self.use_system_prompt,
            system_template=self.system_template,
            prompt_template=self.prompt_template,
            response_template=self.response_template,
        ).task_execution()

        stop_words = [I18N_DEFAULT.slice("observation")]
        if self.response_template:
            stop_words.append(
                self.response_template.split("{{ .Response }}")[1].strip()
            )

        rpm_limit_fn = (
            self._rpm_controller.check_or_wait if self._rpm_controller else None
        )

        return prompt, stop_words, rpm_limit_fn

    def create_agent_executor(
        self, tools: list[BaseTool] | None = None, task: Task | None = None
    ) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        raw_tools: list[BaseTool] = tools or self.tools or []
        parsed_tools = parse_tools(raw_tools)

        prompt, stop_words, rpm_limit_fn = self._build_execution_prompt(raw_tools)

        if self.agent_executor is not None:
            self._update_executor_parameters(
                task=task,
                tools=parsed_tools,
                raw_tools=raw_tools,
                prompt=prompt,
                stop_words=stop_words,
                rpm_limit_fn=rpm_limit_fn,
            )
        else:
            if not isinstance(self.llm, BaseLLM):
                raise RuntimeError(
                    "LLM must be resolved before creating agent executor."
                )
            self.agent_executor = self.executor_class(
                llm=self.llm,
                task=task,
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
                response_model=(
                    task.response_model or task.output_pydantic or task.output_json
                )
                if task
                else None,
            )

    def _update_executor_parameters(
        self,
        task: Task | None,
        tools: list[CrewStructuredTool],
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
        if self.agent_executor is None:
            raise RuntimeError("Agent executor is not initialized.")

        if task is not None:
            self.agent_executor.task = task
        self.agent_executor.tools = tools
        self.agent_executor.original_tools = raw_tools
        self.agent_executor.prompt = prompt
        if isinstance(self.agent_executor, AgentExecutor):
            self.agent_executor.stop_words = stop_words
        else:
            self.agent_executor.stop = stop_words
        self.agent_executor.tools_names = get_tool_names(tools)
        self.agent_executor.tools_description = render_text_description_and_args(tools)
        self.agent_executor.response_model = (
            (task.response_model or task.output_pydantic or task.output_json)
            if task
            else None
        )

        self.agent_executor.tools_handler = self.tools_handler
        self.agent_executor.request_within_rpm_limit = rpm_limit_fn

        if isinstance(self.agent_executor.llm, BaseLLM):
            existing_stop = getattr(self.agent_executor.llm, "stop", [])
            self.agent_executor.llm.stop = list(
                set(
                    existing_stop + stop_words
                    if isinstance(existing_stop, list)
                    else stop_words
                )
            )

    def get_delegation_tools(self, agents: Sequence[BaseAgent]) -> list[BaseTool]:
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

        Delegates to :class:`~crewai.mcp.tool_resolver.MCPToolResolver`.
        """
        self._cleanup_mcp_clients()
        from crewai.mcp.tool_resolver import MCPToolResolver

        self._mcp_resolver = MCPToolResolver(agent=self, logger=self._logger)
        return self._mcp_resolver.resolve(mcps)

    def _cleanup_mcp_clients(self) -> None:
        """Cleanup MCP client connections after task execution."""
        if self._mcp_resolver is not None:
            self._mcp_resolver.cleanup()
            self._mcp_resolver = None

    @staticmethod
    def get_multimodal_tools() -> Sequence[BaseTool]:
        """Return tools for multimodal agent capabilities."""
        from crewai.tools.agent_tools.add_image_tool import AddImageTool

        return [AddImageTool()]

    def get_code_execution_tools(self) -> list[Any]:
        """Deprecated: CodeInterpreterTool is no longer available."""
        warnings.warn(
            "CodeInterpreterTool is no longer available. "
            "Use dedicated sandbox services like E2B or Modal.",
            DeprecationWarning,
            stacklevel=2,
        )
        return []

    @staticmethod
    def get_output_converter(
        llm: BaseLLM, text: str, model: type[BaseModel], instructions: str
    ) -> Converter:
        """Create a Converter instance for transforming LLM output to a structured model."""
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
        """Deprecated: No-op. CodeInterpreterTool is no longer available."""
        warnings.warn(
            "CodeInterpreterTool is no longer available. "
            "Use dedicated sandbox services like E2B or Modal.",
            DeprecationWarning,
            stacklevel=2,
        )
        return

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
        """Set the agent's security fingerprint."""
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
        query = I18N_DEFAULT.slice("knowledge_search_query").format(
            task_prompt=task_prompt
        )
        rewriter_prompt = I18N_DEFAULT.slice("knowledge_search_query_system_prompt")
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
            messages: list[LLMMessage] = [
                {"role": "system", "content": rewriter_prompt},
                {"role": "user", "content": query},
            ]
            rewritten_query = self.llm.call(messages)
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
        if self.apps:
            platform_tools = self.get_platform_tools(self.apps)
            if platform_tools:
                if self.tools is None:
                    self.tools = []
                self.tools.extend(platform_tools)
        if self.mcps:
            mcps = self.get_mcp_tools(self.mcps)
            if mcps:
                if self.tools is None:
                    self.tools = []
                self.tools.extend(mcps)

        raw_tools: list[BaseTool] = self.tools or []

        agent_memory = getattr(self, "memory", None)
        if agent_memory is not None:
            from crewai.tools.memory_tools import create_memory_tools

            existing_names = {sanitize_tool_name(t.name) for t in raw_tools}
            raw_tools.extend(
                mt
                for mt in create_memory_tools(agent_memory)
                if sanitize_tool_name(mt.name) not in existing_names
            )

        parsed_tools = parse_tools(raw_tools)

        agent_info = {
            "id": self.id,
            "role": self.role,
            "goal": self.goal,
            "backstory": self.backstory,
            "tools": raw_tools,
            "verbose": self.verbose,
        }

        prompt, stop_words, rpm_limit_fn = self._build_execution_prompt(raw_tools)

        if _is_resuming_agent_executor(self.agent_executor):
            executor = self.agent_executor
            executor.tools = parsed_tools
            executor.tools_names = get_tool_names(parsed_tools)
            executor.tools_description = render_text_description_and_args(parsed_tools)
            executor.original_tools = raw_tools
            executor.prompt = prompt
            executor.response_model = response_format
            executor.stop_words = stop_words
            executor.tools_handler = self.tools_handler
            executor.step_callback = self.step_callback
            executor.function_calling_llm = cast(
                BaseLLM | None, self.function_calling_llm
            )
            executor.respect_context_window = self.respect_context_window
            executor.request_within_rpm_limit = rpm_limit_fn
            executor.callbacks = [TokenCalcHandler(self._token_process)]
        else:
            executor = AgentExecutor(
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

        if agent_memory is not None:
            try:
                crewai_event_bus.emit(
                    self,
                    event=MemoryRetrievalStartedEvent(
                        task_id=None,
                        source_type="agent_kickoff",
                        from_agent=self,
                    ),
                )
                start_time = time.time()
                matches = agent_memory.recall(formatted_messages, limit=20)
                memory_block = ""
                if matches:
                    memory_block = "Relevant memories:\n" + "\n".join(
                        m.format() for m in matches
                    )
                if memory_block:
                    formatted_messages += "\n\n" + I18N_DEFAULT.slice("memory").format(
                        memory=memory_block
                    )
                crewai_event_bus.emit(
                    self,
                    event=MemoryRetrievalCompletedEvent(
                        task_id=None,
                        memory_content=memory_block,
                        retrieval_time_ms=(time.time() - start_time) * 1000,
                        source_type="agent_kickoff",
                        from_agent=self,
                    ),
                )
            except Exception as e:
                crewai_event_bus.emit(
                    self,
                    event=MemoryRetrievalFailedEvent(
                        task_id=None,
                        source_type="agent_kickoff",
                        from_agent=self,
                        error=str(e),
                    ),
                )

        formatted_messages = append_skill_context(self, formatted_messages)

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
        from_checkpoint: CheckpointConfig | None = None,
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
            from_checkpoint: Optional checkpoint config. If ``restore_from``
                is set, the agent resumes from that checkpoint. Remaining
                config fields enable checkpointing for the run.

        Returns:
            LiteAgentOutput: The result of the agent execution.
            When inside a Flow, returns a coroutine that resolves to LiteAgentOutput.

        Note:
            For explicit async usage outside of Flow, use kickoff_async() directly.
        """
        restored = apply_checkpoint(self, from_checkpoint)
        if restored is not None:
            return restored.kickoff(  # type: ignore[no-any-return]
                messages=messages,
                response_format=response_format,
                input_files=input_files,
            )

        if is_inside_event_loop():
            return self.kickoff_async(messages, response_format, input_files)

        executor, inputs, agent_info, parsed_tools = self._prepare_kickoff(
            messages, response_format, input_files
        )

        try:
            if self.checkpoint_kickoff_event_id is not None:
                self._kickoff_event_id = self.checkpoint_kickoff_event_id
                self.checkpoint_kickoff_event_id = None
            else:
                started_event = LiteAgentExecutionStartedEvent(
                    agent_info=agent_info,
                    tools=parsed_tools,
                    messages=messages,
                )
                crewai_event_bus.emit(self, event=started_event)
                self._kickoff_event_id = started_event.event_id

            output = self._execute_and_build_output(executor, inputs, response_format)
            return self._finalize_kickoff(
                output, executor, inputs, response_format, messages, agent_info
            )

        except Exception as e:
            self._emit_kickoff_error(agent_info, e)

    def _finalize_kickoff(
        self,
        output: LiteAgentOutput,
        executor: AgentExecutor,
        inputs: dict[str, str],
        response_format: type[Any] | None,
        messages: str | list[LLMMessage],
        agent_info: dict[str, Any],
    ) -> LiteAgentOutput:
        """Apply guardrails, save to memory, and emit completion event.

        Args:
            output: The execution output.
            executor: The agent executor.
            inputs: The execution inputs.
            response_format: Optional response format.
            messages: The original messages.
            agent_info: Agent metadata for events.

        Returns:
            The finalized output.
        """
        if self.guardrail is not None:
            output = self._process_kickoff_guardrail(
                output=output,
                executor=executor,
                inputs=inputs,
                response_format=response_format,
            )

        self._save_kickoff_to_memory(messages, output.raw)

        crewai_event_bus.emit(
            self,
            event=LiteAgentExecutionCompletedEvent(
                agent_info=agent_info,
                output=output.raw,
            ),
        )

        return output

    def _emit_kickoff_error(self, agent_info: dict[str, Any], e: Exception) -> NoReturn:
        """Emit a kickoff error event and re-raise."""
        crewai_event_bus.emit(
            self,
            event=LiteAgentExecutionErrorEvent(
                agent_info=agent_info,
                error=str(e),
            ),
        )
        raise e

    def _save_kickoff_to_memory(
        self, messages: str | list[LLMMessage], output_text: str
    ) -> None:
        """Save kickoff result to memory. No-op if agent has no memory."""
        agent_memory = getattr(self, "memory", None)
        if agent_memory is None:
            return
        try:
            if isinstance(messages, str):
                input_str = messages
            else:
                input_str = (
                    "\n".join(
                        str(msg.get("content", ""))
                        for msg in messages
                        if msg.get("content")
                    )
                    or "User request"
                )
            raw = f"Input: {input_str}\nAgent: {self.role}\nResult: {output_text}"
            extracted = agent_memory.extract_memories(raw)
            if extracted:
                agent_memory.remember_many(extracted)
        except Exception as e:
            self._logger.log("error", f"Failed to save kickoff result to memory: {e}")

    def _build_output_from_result(
        self,
        result: dict[str, Any],
        executor: AgentExecutor,
        response_format: type[Any] | None = None,
    ) -> LiteAgentOutput:
        """Build a LiteAgentOutput from an executor result dict.

        Shared logic used by both sync and async execution paths.

        Args:
            result: The result dictionary from executor.invoke / invoke_async.
            executor: The executor instance.
            response_format: Optional response format.

        Returns:
            LiteAgentOutput with raw output, formatted result, and metrics.
        """
        output = result.get("output", "")

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
                instructions = I18N_DEFAULT.slice("formatted_task_instructions").format(
                    output_format=schema
                )

                converter = Converter(
                    llm=cast(BaseLLM, self.llm),
                    text=raw_output,
                    model=response_format,
                    instructions=instructions,
                )

                conversion_result = converter.to_pydantic()
                if isinstance(conversion_result, BaseModel):
                    formatted_result = conversion_result
            except ConverterError:
                pass
        else:
            raw_output = str(output) if not isinstance(output, str) else output

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

        todo_results = LiteAgentOutput.from_todo_items(executor.state.todos.items)

        return LiteAgentOutput(
            raw=raw_str,
            pydantic=formatted_result,
            agent_role=self.role,
            usage_metrics=usage_metrics.model_dump() if usage_metrics else None,
            messages=list(executor.state.messages),
            plan=executor.state.plan,
            todos=todo_results,
            replan_count=executor.state.replan_count,
            last_replan_reason=executor.state.last_replan_reason,
        )

    def _execute_and_build_output(
        self,
        executor: AgentExecutor,
        inputs: dict[str, str],
        response_format: type[Any] | None = None,
    ) -> LiteAgentOutput:
        """Execute the agent synchronously and build the output object."""
        result = cast(dict[str, Any], executor.invoke(inputs))
        return self._build_output_from_result(result, executor, response_format)

    async def _execute_and_build_output_async(
        self,
        executor: AgentExecutor,
        inputs: dict[str, str],
        response_format: type[Any] | None = None,
    ) -> LiteAgentOutput:
        """Execute the agent asynchronously and build the output object."""
        result = await executor.invoke_async(inputs)
        return self._build_output_from_result(result, executor, response_format)

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

            executor._append_message_to_state(
                guardrail_result.error or "Guardrail validation failed",
                role="user",
            )

            output = self._execute_and_build_output(executor, inputs, response_format)

            return self._process_kickoff_guardrail(
                output=output,
                executor=executor,
                inputs=inputs,
                response_format=response_format,
                retry_count=retry_count + 1,
            )

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
        from_checkpoint: CheckpointConfig | None = None,
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
            from_checkpoint: Optional checkpoint config. If ``restore_from``
                is set, the agent resumes from that checkpoint.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        restored = apply_checkpoint(self, from_checkpoint)
        if restored is not None:
            return await restored.kickoff_async(  # type: ignore[no-any-return]
                messages=messages,
                response_format=response_format,
                input_files=input_files,
            )

        executor, inputs, agent_info, parsed_tools = self._prepare_kickoff(
            messages, response_format, input_files
        )

        try:
            if self.checkpoint_kickoff_event_id is not None:
                self._kickoff_event_id = self.checkpoint_kickoff_event_id
                self.checkpoint_kickoff_event_id = None
            else:
                started_event = LiteAgentExecutionStartedEvent(
                    agent_info=agent_info,
                    tools=parsed_tools,
                    messages=messages,
                )
                crewai_event_bus.emit(self, event=started_event)
                self._kickoff_event_id = started_event.event_id

            output = await self._execute_and_build_output_async(
                executor, inputs, response_format
            )
            return self._finalize_kickoff(
                output, executor, inputs, response_format, messages, agent_info
            )

        except Exception as e:
            self._emit_kickoff_error(agent_info, e)

    async def akickoff(
        self,
        messages: str | list[LLMMessage],
        response_format: type[Any] | None = None,
        input_files: dict[str, FileInput] | None = None,
        from_checkpoint: CheckpointConfig | None = None,
    ) -> LiteAgentOutput:
        """Async version of kickoff. Alias for kickoff_async.

        Args:
            messages: Either a string query or a list of message dictionaries.
            response_format: Optional Pydantic model for structured output.
            input_files: Optional dict of named files to attach to the message.
            from_checkpoint: Optional checkpoint config. If ``restore_from``
                is set, the agent resumes from that checkpoint.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        return await self.kickoff_async(
            messages, response_format, input_files, from_checkpoint
        )
