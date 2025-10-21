from __future__ import annotations

import asyncio
from collections.abc import Sequence
import shutil
import subprocess
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
)

from pydantic import BaseModel, Field, InstanceOf, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.knowledge_events import (
    KnowledgeQueryCompletedEvent,
    KnowledgeQueryFailedEvent,
    KnowledgeQueryStartedEvent,
    KnowledgeRetrievalCompletedEvent,
    KnowledgeRetrievalStartedEvent,
    KnowledgeSearchQueryFailedEvent,
)
from crewai.events.types.memory_events import (
    MemoryRetrievalCompletedEvent,
    MemoryRetrievalStartedEvent,
)
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.utils.knowledge_utils import extract_knowledge_context
from crewai.lite_agent import LiteAgent
from crewai.llms.base_llm import BaseLLM
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.rag.embeddings.types import EmbedderConfig
from crewai.security.fingerprint import Fingerprint
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities.agent_utils import (
    get_tool_names,
    load_agent_from_repository,
    parse_tools,
    render_text_description_and_args,
)
from crewai.utilities.constants import TRAINED_AGENTS_DATA_FILE, TRAINING_DATA_FILE
from crewai.utilities.converter import Converter, generate_model_description
from crewai.utilities.guardrail_types import GuardrailType
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.prompts import Prompts
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.training_handler import CrewTrainingHandler


if TYPE_CHECKING:
    from crewai_tools import CodeInterpreterTool

    from crewai.agents.agent_builder.base_agent import PlatformAppOrAction
    from crewai.lite_agent_output import LiteAgentOutput
    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


# MCP Connection timeout constants (in seconds)
MCP_CONNECTION_TIMEOUT = 10
MCP_TOOL_EXECUTION_TIMEOUT = 30
MCP_DISCOVERY_TIMEOUT = 15
MCP_MAX_RETRIES = 3

# Simple in-memory cache for MCP tool schemas (duration: 5 minutes)
_mcp_schema_cache = {}
_cache_ttl = 300  # 5 minutes


class Agent(BaseAgent):
    """Represents an agent in a system.

    Each agent has a role, a goal, a backstory, and an optional language model (llm).
    The agent can also have memory, can operate in verbose mode, and can delegate tasks to other agents.

    Attributes:
            agent_executor: An instance of the CrewAgentExecutor class.
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

    _times_executed: int = PrivateAttr(default=0)
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
        description="Whether the agent is multimodal.",
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
        if self.reasoning:
            try:
                from crewai.utilities.reasoning_handler import (
                    AgentReasoning,
                    AgentReasoningOutput,
                )

                reasoning_handler = AgentReasoning(task=task, agent=self)
                reasoning_output: AgentReasoningOutput = (
                    reasoning_handler.handle_agent_reasoning()
                )

                # Add the reasoning plan to the task description
                task.description += f"\n\nReasoning Plan:\n{reasoning_output.plan.plan}"
            except Exception as e:
                self._logger.log("error", f"Error during reasoning process: {e!s}")
        self._inject_date_to_task(task)

        if self.tools_handler:
            self.tools_handler.last_used_tool = None

        task_prompt = task.prompt()

        # If the task requires output in JSON or Pydantic format,
        # append specific instructions to the task prompt to ensure
        # that the final answer does not include any code block markers
        if task.output_json or task.output_pydantic:
            # Generate the schema based on the output format
            if task.output_json:
                # schema = json.dumps(task.output_json, indent=2)
                schema = generate_model_description(task.output_json)
                task_prompt += "\n" + self.i18n.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

            elif task.output_pydantic:
                schema = generate_model_description(task.output_pydantic)
                task_prompt += "\n" + self.i18n.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

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
        knowledge_config = (
            self.knowledge_config.model_dump() if self.knowledge_config else {}
        )

        if self.knowledge or (self.crew and self.crew.knowledge):
            crewai_event_bus.emit(
                self,
                event=KnowledgeRetrievalStartedEvent(
                    from_task=task,
                    from_agent=self,
                ),
            )
            try:
                self.knowledge_search_query = self._get_knowledge_search_query(
                    task_prompt, task
                )
                if self.knowledge_search_query:
                    # Quering agent specific knowledge
                    if self.knowledge:
                        agent_knowledge_snippets = self.knowledge.query(
                            [self.knowledge_search_query], **knowledge_config
                        )
                        if agent_knowledge_snippets:
                            self.agent_knowledge_context = extract_knowledge_context(
                                agent_knowledge_snippets
                            )
                            if self.agent_knowledge_context:
                                task_prompt += self.agent_knowledge_context

                    # Quering crew specific knowledge
                    knowledge_snippets = self.crew.query_knowledge(
                        [self.knowledge_search_query], **knowledge_config
                    )
                    if knowledge_snippets:
                        self.crew_knowledge_context = extract_knowledge_context(
                            knowledge_snippets
                        )
                        if self.crew_knowledge_context:
                            task_prompt += self.crew_knowledge_context

                    crewai_event_bus.emit(
                        self,
                        event=KnowledgeRetrievalCompletedEvent(
                            query=self.knowledge_search_query,
                            from_task=task,
                            from_agent=self,
                            retrieved_knowledge=(
                                (self.agent_knowledge_context or "")
                                + (
                                    "\n"
                                    if self.agent_knowledge_context
                                    and self.crew_knowledge_context
                                    else ""
                                )
                                + (self.crew_knowledge_context or "")
                            ),
                        ),
                    )
            except Exception as e:
                crewai_event_bus.emit(
                    self,
                    event=KnowledgeSearchQueryFailedEvent(
                        query=self.knowledge_search_query or "",
                        error=str(e),
                        from_task=task,
                        from_agent=self,
                    ),
                )

        tools = tools or self.tools or []
        self.create_agent_executor(tools=tools, task=task)

        if self.crew and self.crew._train:
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

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

            # Determine execution method based on timeout setting
            if self.max_execution_time is not None:
                if (
                    not isinstance(self.max_execution_time, int)
                    or self.max_execution_time <= 0
                ):
                    raise ValueError(
                        "Max Execution time must be a positive integer greater than zero"
                    )
                result = self._execute_with_timeout(
                    task_prompt, task, self.max_execution_time
                )
            else:
                result = self._execute_without_timeout(task_prompt, task)

        except TimeoutError as e:
            # Propagate TimeoutError without retry
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
                # Do not retry on litellm errors
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

        # If there was any tool in self.tools_results that had result_as_answer
        # set to True, return the results of the last tool that had
        # result_as_answer set to True
        for tool_result in self.tools_results:
            if tool_result.get("result_as_answer", False):
                result = tool_result["result"]
        crewai_event_bus.emit(
            self,
            event=AgentExecutionCompletedEvent(agent=self, task=task, output=result),
        )
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

    def create_agent_executor(
        self, tools: list[BaseTool] | None = None, task: Task | None = None
    ) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        raw_tools: list[BaseTool] = tools or self.tools or []
        parsed_tools = parse_tools(raw_tools)

        prompt = Prompts(
            agent=self,
            has_tools=len(raw_tools) > 0,
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

        self.agent_executor = CrewAgentExecutor(
            llm=self.llm,
            task=task,  # type: ignore[arg-type]
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
            request_within_rpm_limit=(
                self._rpm_controller.check_or_wait if self._rpm_controller else None
            ),
            callbacks=[TokenCalcHandler(self._token_process)],
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

    def get_mcp_tools(self, mcps: list[str]) -> list[BaseTool]:
        """Convert MCP server references to CrewAI tools."""
        all_tools = []

        for mcp_ref in mcps:
            try:
                if mcp_ref.startswith("crewai-amp:"):
                    tools = self._get_amp_mcp_tools(mcp_ref)
                elif mcp_ref.startswith("https://"):
                    tools = self._get_external_mcp_tools(mcp_ref)
                else:
                    continue

                all_tools.extend(tools)
                self._logger.log(
                    "info", f"Successfully loaded {len(tools)} tools from {mcp_ref}"
                )

            except Exception as e:
                self._logger.log("warning", f"Skipping MCP {mcp_ref} due to error: {e}")
                continue

        return all_tools

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

            return tools

        except Exception as e:
            self._logger.log(
                "warning", f"Failed to connect to MCP server {server_url}: {e}"
            )
            return []

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

    def _extract_server_name(self, server_url: str) -> str:
        """Extract clean server name from URL for tool prefixing."""
        from urllib.parse import urlparse

        parsed = urlparse(server_url)
        domain = parsed.netloc.replace(".", "_")
        path = parsed.path.replace("/", "_").strip("_")
        return f"{domain}_{path}" if path else domain

    def _get_mcp_tool_schemas(self, server_params: dict) -> dict[str, dict]:
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
                return cached_data

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

    async def _get_mcp_tool_schemas_async(self, server_params: dict) -> dict[str, dict]:
        """Async implementation of MCP tool schema retrieval with timeouts and retries."""
        server_url = server_params["url"]
        return await self._retry_mcp_discovery(
            self._discover_mcp_tools_with_timeout, server_url
        )

    async def _retry_mcp_discovery(
        self, operation_func, server_url: str
    ) -> dict[str, dict]:
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

    async def _attempt_mcp_discovery(
        self, operation_func, server_url: str
    ) -> tuple[dict[str, dict] | None, str, bool]:
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
    ) -> dict[str, dict]:
        """Discover MCP tools with timeout wrapper."""
        return await asyncio.wait_for(
            self._discover_mcp_tools(server_url), timeout=MCP_DISCOVERY_TIMEOUT
        )

    async def _discover_mcp_tools(self, server_url: str) -> dict[str, dict]:
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
                            tool.name, tool.inputSchema
                        )

                    schemas[tool.name] = {
                        "description": getattr(tool, "description", ""),
                        "args_schema": args_schema,
                    }
                return schemas

    def _json_schema_to_pydantic(self, tool_name: str, json_schema: dict) -> type:
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

        field_definitions = {}

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
        return create_model(model_name, **field_definitions)

    def _json_type_to_python(self, field_schema: dict) -> type:
        """Convert JSON Schema type to Python type.

        Args:
            field_schema: JSON Schema field definition

        Returns:
            Python type
        """
        from typing import Any

        json_type = field_schema.get("type")

        if "anyOf" in field_schema:
            types = []
            for option in field_schema["anyOf"]:
                if "const" in option:
                    types.append(str)
                else:
                    types.append(self._json_type_to_python(option))
            unique_types = list(set(types))
            if len(unique_types) > 1:
                result = unique_types[0]
                for t in unique_types[1:]:
                    result = result | t
                return result
            return unique_types[0]

        type_mapping = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        return type_mapping.get(json_type, Any)

    def _fetch_amp_mcp_servers(self, mcp_name: str) -> list[dict]:
        """Fetch MCP server configurations from CrewAI AMP API."""
        # TODO: Implement AMP API call to "integrations/mcps" endpoint
        # Should return list of server configs with URLs
        return []

    def get_multimodal_tools(self) -> Sequence[BaseTool]:
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

    def get_output_converter(
        self, llm: BaseLLM, text: str, model: type[BaseModel], instructions: str
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

    def _render_text_description(self, tools: list[Any]) -> str:
        """Render the tool name and description in plain text.

        Output will be in the format of:

        .. code-block:: markdown

            search: This tool is used for search
            calculator: This tool is used for math
        """
        return "\n".join(
            [
                f"Tool name: {tool.name}\nTool description:\n{tool.description}"
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

    def kickoff(
        self,
        messages: str | list[LLMMessage],
        response_format: type[Any] | None = None,
    ) -> LiteAgentOutput:
        """
        Execute the agent with the given messages using a LiteAgent instance.

        This method is useful when you want to use the Agent configuration but
        with the simpler and more direct execution flow of LiteAgent.

        Args:
            messages: Either a string query or a list of message dictionaries.
                     If a string is provided, it will be converted to a user message.
                     If a list is provided, each dict should have 'role' and 'content' keys.
            response_format: Optional Pydantic model for structured output.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        lite_agent = LiteAgent(
            id=self.id,
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            llm=self.llm,
            tools=self.tools or [],
            max_iterations=self.max_iter,
            max_execution_time=self.max_execution_time,
            respect_context_window=self.respect_context_window,
            verbose=self.verbose,
            response_format=response_format,
            i18n=self.i18n,
            original_agent=self,
            guardrail=self.guardrail,
            guardrail_max_retries=self.guardrail_max_retries,
        )

        return lite_agent.kickoff(messages)

    async def kickoff_async(
        self,
        messages: str | list[LLMMessage],
        response_format: type[Any] | None = None,
    ) -> LiteAgentOutput:
        """
        Execute the agent asynchronously with the given messages using a LiteAgent instance.

        This is the async version of the kickoff method.

        Args:
            messages: Either a string query or a list of message dictionaries.
                     If a string is provided, it will be converted to a user message.
                     If a list is provided, each dict should have 'role' and 'content' keys.
            response_format: Optional Pydantic model for structured output.

        Returns:
            LiteAgentOutput: The result of the agent execution.
        """
        lite_agent = LiteAgent(
            role=self.role,
            goal=self.goal,
            backstory=self.backstory,
            llm=self.llm,
            tools=self.tools or [],
            max_iterations=self.max_iter,
            max_execution_time=self.max_execution_time,
            respect_context_window=self.respect_context_window,
            verbose=self.verbose,
            response_format=response_format,
            i18n=self.i18n,
            original_agent=self,
            guardrail=self.guardrail,
            guardrail_max_retries=self.guardrail_max_retries,
        )

        return await lite_agent.kickoff_async(messages)
