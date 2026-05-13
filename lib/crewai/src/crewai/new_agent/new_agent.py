"""NewAgent — standalone, conversational, self-improving agent."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
import importlib.util
import logging
from pathlib import Path
import re
import threading
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.new_agent.models import (
    AgentSettings,
    MemoryScope,
    MemorySlice,
    Message,
    PromptStack,
    ProvenanceEntry,
)
from crewai.new_agent.provider import DirectProvider


logger = logging.getLogger(__name__)


# ── GAP-56: Circular coworker guard ─────────────────────────────
_init_chain = threading.local()


def _get_init_chain() -> set[str]:
    """Return the thread-local set of agent IDs currently being initialized."""
    if not hasattr(_init_chain, "agent_ids"):
        _init_chain.agent_ids = set()
    return _init_chain.agent_ids


# ── GAP-63: Process-level AMP definition cache ──────────────────
_amp_cache: dict[str, dict] = {}


def clear_amp_cache() -> None:
    """Clear the process-level AMP coworker definition cache."""
    _amp_cache.clear()


# ── GAP-24: Pronouns that trigger anaphora resolution ───────────
_ANAPHORA_PRONOUNS = re.compile(
    r"\b(he|she|it|they|this|that|these|those)\b",
    re.IGNORECASE,
)


class NewAgent(BaseModel):
    """Standalone conversational agent.

    Replaces the Agent + Task + Crew pattern with a direct
    message-based interface: message(), amessage(), stream().
    """

    model_config = {"arbitrary_types_allowed": True}

    # Identity
    id: str = Field(default_factory=lambda: uuid4().hex)
    role: str
    goal: str
    backstory: str = ""

    # LLM
    llm: str | Any | None = None
    function_calling_llm: str | Any | None = None

    # Capabilities
    tools: list[Any] = Field(default_factory=list)
    skills: list[Any] = Field(default_factory=list)
    mcps: list[Any] = Field(default_factory=list)
    apps: list[Any] = Field(default_factory=list)

    # Collaboration
    coworkers: list[Any] = Field(default_factory=list)

    # Knowledge & Memory
    knowledge: Any | None = None
    knowledge_sources: list[Any] = Field(default_factory=list)
    memory: bool | Any = True

    # Settings
    settings: AgentSettings = Field(default_factory=AgentSettings)

    # Execution
    max_iter: int = 25
    max_tokens: int | None = None
    max_execution_time: int | None = None
    verbose: bool = False

    # Guardrails
    guardrail: Any | None = None

    # Structured output
    response_model: type[BaseModel] | None = None

    # Self-construction from AMP repository
    from_repository: str | None = None

    # Security & A2A
    security_config: Any | None = None
    a2a: Any | None = None

    # Hooks
    on_message: Callable[..., Any] | None = Field(default=None, exclude=True)
    on_delegate: Callable[..., Any] | None = Field(default=None, exclude=True)
    on_complete: Callable[..., Any] | None = Field(default=None, exclude=True)
    step_callback: Callable[..., Any] | None = Field(default=None, exclude=True)

    # Provider (transport) — typed as Any to allow duck-typed providers and mocks.
    # Implements the ConversationalProvider protocol from crewai.new_agent.provider.
    provider: Any | None = Field(default=None, exclude=True)

    # GAP-41: Manual memory scope override
    memory_scope: str | None = None

    # Private
    _llm_instance: Any = PrivateAttr(default=None)
    _memory_instance: Any = PrivateAttr(default=None)
    _resolved_tools: list[Any] = PrivateAttr(default_factory=list)
    _coworker_tools: list[Any] = PrivateAttr(default_factory=list)
    _resolved_coworkers: list[Any] = PrivateAttr(default_factory=list)
    # GAP-31: Concurrent conversation support — dict of executors keyed by conversation_id
    _executors: dict[str, Any] = PrivateAttr(default_factory=dict)
    _default_conversation_id: str = PrivateAttr(default_factory=lambda: uuid4().hex)
    _dreaming_engine: Any = PrivateAttr(default=None)
    _planning_engine: Any = PrivateAttr(default=None)
    _knowledge_discovery: Any = PrivateAttr(default=None)
    _skill_builder: Any = PrivateAttr(default=None)
    _active_skills: list[Any] = PrivateAttr(default_factory=list)
    _telemetry: Any = PrivateAttr(default=None)
    _conversation_id: str = PrivateAttr(default_factory=lambda: uuid4().hex)
    _logger: logging.Logger = PrivateAttr(
        default_factory=lambda: logging.getLogger("crewai.new_agent")
    )
    # GAP-41/45: Memory namespace and filter from MemoryScope/MemorySlice
    _memory_namespace: str | None = PrivateAttr(default=None)
    _memory_shared: bool = PrivateAttr(default=False)
    _memory_filter: Any = PrivateAttr(default=None)
    # GAP-38: Stored A2A configuration
    _a2a_config: Any = PrivateAttr(default=None)
    # GAP-31: Provider instance for creating new executors
    _provider: Any = PrivateAttr(default=None)
    # GAP-86: Flag indicating agent was resolved from AMP repository
    _amp_resolved: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _load_from_repository(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("from_repository"):
            handle = data["from_repository"]
            try:
                from crewai.utilities.agent_utils import load_agent_from_repository

                attrs = load_agent_from_repository(handle)
                for key, val in attrs.items():
                    if key not in data or data[key] is None:
                        data[key] = val
            except Exception:
                pass
        return data

    @model_validator(mode="after")
    def _setup(self) -> Self:
        """Initialize LLM, tools, coworkers, and executor."""
        self._init_llm()
        self._init_memory()
        self._init_tools()
        self._init_skills()
        self._init_apps_warning()
        self._init_security_a2a()

        # GAP-56: Circular coworker guard
        chain = _get_init_chain()
        if self.id in chain:
            # GAP-99: Log a clear warning when circular coworker reference is detected
            logger.warning(
                f"Circular coworker reference detected for agent '{self.role}' (id={self.id}). "
                f"Skipping coworker initialization to prevent infinite recursion. "
                f"Check your coworker configuration."
            )
            self._init_engines()
            self._init_telemetry()
            self._init_executor()
            self._emit_created_event()
            return self

        chain.add(self.id)
        try:
            self._init_coworkers()
        finally:
            chain.discard(self.id)

        self._init_engines()
        self._init_telemetry()
        self._init_executor()
        self._emit_created_event()
        return self

    def _init_llm(self) -> None:
        from crewai.utilities.llm_utils import create_llm

        self._llm_instance = create_llm(self.llm)
        if self._llm_instance is None:
            self._llm_instance = create_llm(None)

    def _init_memory(self) -> None:
        """Initialize memory if enabled.

        GAP-45: Accepts MemoryScope and MemorySlice as memory field values.
        GAP-41: Reads memory_scope from provider context or manual override.
        """
        if not self.settings.memory_enabled:
            self._memory_instance = None
            return

        if self.memory is False:
            self._memory_instance = None
            return

        # GAP-45: Handle MemoryScope / MemorySlice types
        if isinstance(self.memory, MemoryScope):
            self._memory_namespace = self.memory.namespace
            self._memory_shared = self.memory.shared
            self._init_memory_instance()
            return

        if isinstance(self.memory, MemorySlice):
            self._memory_namespace = self.memory.scope or None
            self._memory_filter = self.memory
            self._init_memory_instance()
            return

        try:
            from crewai.memory.unified_memory import Memory
            from crewai.memory.utils import sanitize_scope_name

            if isinstance(self.memory, Memory):
                self._memory_instance = self.memory
            elif self.memory is True or self.memory is None:
                agent_name = sanitize_scope_name(self.role or str(self.id))
                self._memory_instance = Memory(root_scope=f"/agent/{agent_name}")
            else:
                self._memory_instance = self.memory
        except Exception as e:
            self._logger.debug(f"Memory initialization failed: {e}")
            self._memory_instance = None

        if self._memory_instance and self.settings.memory_read_only:
            self._memory_instance.read_only = True

        # GAP-41: Apply memory scope from provider or manual override
        scope = self.memory_scope
        if scope is None:
            provider = self.provider
            if provider is not None:
                scope = getattr(provider, "memory_scope", None)
        if scope:
            self._memory_namespace = scope

    def _init_memory_instance(self) -> None:
        """Create a Memory instance (used by MemoryScope/MemorySlice paths)."""
        try:
            from crewai.memory.unified_memory import Memory
            from crewai.memory.utils import sanitize_scope_name

            agent_name = sanitize_scope_name(self.role or str(self.id))
            self._memory_instance = Memory(root_scope=f"/agent/{agent_name}")
        except Exception as e:
            self._logger.debug(f"Memory initialization failed: {e}")
            self._memory_instance = None

    def _init_tools(self) -> None:
        """Resolve tools from various sources."""
        resolved: list[Any] = []

        for tool in self.tools:
            resolved.append(tool)

        if self.mcps:
            try:
                from crewai.mcp.tool_resolver import MCPToolResolver

                resolver = MCPToolResolver(agent=self, logger=self._logger)
                mcp_tools = resolver.resolve(self.mcps)
                resolved.extend(mcp_tools)
            except Exception as e:
                self._logger.warning(f"Failed to resolve MCP tools: {e}")

        self._resolved_tools = resolved

        if getattr(self.settings, "can_schedule", False):
            try:
                from crewai.new_agent.scheduler import ScheduleTaskTool

                agent_name = getattr(self, "role", "") or str(self.id)
                self._resolved_tools.append(ScheduleTaskTool(agent_name=agent_name))
            except Exception:
                pass

    def _init_skills(self) -> None:
        """Resolve skills from Path objects into SKILL.md-based Skill instances,
        falling back to Python module loading for backward compatibility."""
        if not self.skills:
            return

        for skill in self.skills:
            if isinstance(skill, (str, Path)):
                skill_path = Path(skill)
                if skill_path.is_dir() and (skill_path / "SKILL.md").exists():
                    try:
                        from crewai.skills.loader import activate_skill, discover_skills

                        discovered = discover_skills(skill_path.parent)
                        for s in discovered:
                            if s.name == skill_path.name:
                                activated = activate_skill(s)
                                self._active_skills.append(activated)
                    except Exception as e:
                        self._logger.warning(
                            f"Failed to load SKILL.md from {skill_path}: {e}"
                        )
                else:
                    self._load_python_skill(skill_path)
            elif hasattr(skill, "run") or hasattr(skill, "_run"):
                self._resolved_tools.append(skill)
            else:
                try:
                    from crewai.skills.models import Skill as SkillModel

                    if isinstance(skill, SkillModel):
                        self._active_skills.append(skill)
                except Exception:
                    pass

    def _load_python_skill(self, skill_path: Path) -> None:
        """Load a Python module as tool instances (backward compatibility)."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"skill_{skill_path.stem}",
                str(skill_path),
            )
            if spec is None or spec.loader is None:
                self._logger.warning(f"Cannot load skill from {skill_path}")
                return
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and attr_name != "BaseTool"
                    and hasattr(attr, "run")
                ):
                    try:
                        self._resolved_tools.append(attr())
                    except Exception:
                        pass
        except Exception as e:
            self._logger.warning(f"Failed to load skill from {skill_path}: {e}")

    def _init_apps_warning(self) -> None:
        """GAP-36: Log a warning when apps are specified (platform-managed)."""
        if self.apps:
            self._logger.warning(
                "Apps integration requires the CrewAI Platform. "
                f"{len(self.apps)} app(s) configured but not resolved locally."
            )

    def _init_security_a2a(self) -> None:
        """GAP-38: Store security_config and a2a fields for later use."""
        if self.security_config is not None:
            self._logger.info(
                f"Security configuration applied: {type(self.security_config).__name__}"
            )

        if self.a2a is not None:
            self._a2a_config = self.a2a
            self._logger.info(
                "A2A server configured — agent will be accessible via A2A protocol"
            )

    def _init_coworkers(self) -> None:
        """Resolve coworker references into delegation tools."""
        from crewai.new_agent.coworker_tools import build_coworker_tools

        self._resolved_coworkers = []
        self._coworker_tools = []

        for cw in self.coworkers:
            if isinstance(cw, NewAgent):
                if cw.id == self.id or cw.role == self.role:
                    continue
                self._resolved_coworkers.append(cw)
            elif isinstance(cw, str):
                try:
                    resolved = self._resolve_amp_coworker(cw)
                    self._resolved_coworkers.append(resolved)
                except Exception as e:
                    self._logger.warning(f"Failed to resolve AMP coworker '{cw}': {e}")
            elif isinstance(cw, dict):
                # GAP-86: Support both plan format {"amp": "handle"} and legacy {"handle": "handle"}
                handle = cw.get("amp") or cw.get("handle")
                if handle:
                    overrides = {
                        k: v
                        for k, v in cw.items()
                        if k not in ("amp", "handle", "overrides")
                    }
                    overrides.update(cw.get("overrides", {}))
                    try:
                        resolved = self._resolve_amp_coworker(
                            handle,
                            overrides=overrides or None,
                        )
                        resolved._amp_resolved = True
                        self._resolved_coworkers.append(resolved)
                    except Exception as e:
                        self._logger.warning(
                            f"Failed to resolve AMP coworker '{handle}': {e}"
                        )
                else:
                    self._resolved_coworkers.append(cw)
            else:
                self._resolved_coworkers.append(cw)

        if self._resolved_coworkers:
            self._coworker_tools = build_coworker_tools(
                self._resolved_coworkers,
                parent_role=self.role,
                parent_agent=self,
            )

    def _init_engines(self) -> None:
        """Initialize dreaming, planning, knowledge discovery, and skill builder."""
        from crewai.new_agent.dreaming import DreamingEngine
        from crewai.new_agent.knowledge_discovery import KnowledgeDiscovery
        from crewai.new_agent.planning import PlanningEngine

        if self.settings.self_improving:
            self._dreaming_engine = DreamingEngine(self)
        if self.settings.planning_enabled:
            self._planning_engine = PlanningEngine(self)
        self._knowledge_discovery = KnowledgeDiscovery(self)

        if self.settings.can_build_skills:
            try:
                from crewai.new_agent.skill_builder import SkillBuilder

                self._skill_builder = SkillBuilder(self)
            except Exception:
                pass

    def _resolve_amp_coworker(
        self,
        handle: str,
        overrides: dict[str, Any] | None = None,
    ) -> NewAgent:
        """Resolve an AMP repository handle into a NewAgent instance.

        GAP-63: Uses a process-level cache to avoid redundant API calls.
        """
        from crewai.utilities.agent_utils import load_agent_from_repository

        # GAP-63: Check cache first
        if handle in _amp_cache:
            attrs = _amp_cache[handle]
        else:
            attrs = load_agent_from_repository(handle)
            _amp_cache[handle] = attrs

        kwargs: dict[str, Any] = {
            "role": attrs.get("role", handle),
            "goal": attrs.get("goal", ""),
            "backstory": attrs.get("backstory", ""),
            "tools": attrs.get("tools", []),
            "llm": attrs.get("llm", self.llm),
        }
        if overrides:
            for key, val in overrides.items():
                kwargs[key] = val
        return NewAgent(**kwargs)

    def _init_telemetry(self) -> None:
        try:
            from crewai.new_agent.telemetry import NewAgentTelemetry, register_agent

            self._telemetry = NewAgentTelemetry(
                share_data=getattr(self.settings, "share_data", False),
            )
            # GAP-123: Register so event listeners can look up this telemetry instance
            register_agent(self.id, self._telemetry)
            # GAP-124: Compute and set agent fingerprint
            self._telemetry.set_fingerprint(self._compute_fingerprint())
        except Exception:
            pass

    def _compute_fingerprint(self) -> str:
        """GAP-124: Stable hash of agent config for telemetry correlation."""
        import hashlib

        tool_names = sorted(
            getattr(t, "name", "") or getattr(t, "__name__", str(t))
            for t in self._resolved_tools
        )
        parts = [
            self.role,
            self.goal[:100],
            ",".join(tool_names),
            str(self.settings.planning_enabled),
            str(self.settings.self_improving),
        ]
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
        return digest

    def _emit_created_event(self) -> None:
        """GAP-84: Emit agent-created event at construction time.

        The conversation_started event is now emitted in _get_or_create_executor
        when a NEW conversation executor is actually created.
        """
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentCreatedEvent

            crewai_event_bus.emit(
                self,
                NewAgentCreatedEvent(
                    new_agent_id=self.id,
                    new_agent_role=self.role,
                ),
            )
        except Exception:
            pass

        if self._telemetry:
            amp_count = sum(
                1
                for cw in self._resolved_coworkers
                if getattr(cw, "_amp_resolved", False)
            )
            self._telemetry.agent_created(
                agent_id=self.id,
                role=self.role,
                goal=self.goal,
                llm=str(self.llm or ""),
                tools_count=len(self._resolved_tools),
                coworkers_count=len(self._resolved_coworkers),
                memory_enabled=self.settings.memory_enabled,
                planning_enabled=self.settings.planning_enabled,
                coworker_amp_count=amp_count,
            )

    def _init_executor(self) -> None:
        """Create the default executor and store the provider for future use."""
        self._provider = self.provider or DirectProvider()
        executor = self._create_executor(self._provider)
        # GAP-31: Store in the executors dict keyed by default conversation ID
        self._default_conversation_id = self._conversation_id
        self._executors[self._default_conversation_id] = executor

    def _create_executor(self, provider: Any) -> Any:
        """Create a new ConversationalAgentExecutor instance."""
        from crewai.new_agent.executor import ConversationalAgentExecutor

        return ConversationalAgentExecutor(
            agent=self,
            provider=provider,
            max_iter=self.max_iter,
            verbose=self.verbose,
        )

    def _get_or_create_executor(self, conversation_id: str) -> Any:
        """GAP-31: Get an existing executor or create a new one for the given conversation ID.

        New conversations get a fresh DirectProvider so their history is isolated.
        GAP-84: Emits NewAgentConversationStartedEvent when a NEW executor is created.
        """
        if conversation_id in self._executors:
            return self._executors[conversation_id]
        # Create a fresh provider for the new conversation so history is isolated
        executor = self._create_executor(DirectProvider())
        self._executors[conversation_id] = executor

        # GAP-84: Emit conversation_started when a new conversation begins
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentConversationStartedEvent

            crewai_event_bus.emit(
                self,
                NewAgentConversationStartedEvent(
                    conversation_id=conversation_id,
                    new_agent_id=self.id,
                    new_agent_role=self.role,
                ),
            )
        except Exception:
            pass

        return executor

    @property
    def _executor(self) -> Any:
        """Return the default conversation's executor (backward compatibility)."""
        return self._executors.get(self._default_conversation_id)

    # ── Public API ──────────────────────────────────────────────

    def message(
        self, content: str, *, conversation_id: str | None = None, **kwargs: Any
    ) -> Message:
        """Send a message and get a response (sync).

        GAP-31: Accepts optional conversation_id for concurrent conversations.
        """
        cid = conversation_id or self._default_conversation_id
        executor = self._get_or_create_executor(cid)
        user_msg = Message(
            conversation_id=cid,
            role="user",
            content=content,
        )

        if self.on_message:
            self.on_message(user_msg)

        response = executor.invoke(user_msg)

        if self.on_complete:
            self.on_complete(response)

        return response

    async def amessage(
        self, content: str, *, conversation_id: str | None = None, **kwargs: Any
    ) -> Message:
        """Send a message and get a response (async).

        GAP-31: Accepts optional conversation_id for concurrent conversations.
        """
        cid = conversation_id or self._default_conversation_id
        executor = self._get_or_create_executor(cid)
        user_msg = Message(
            conversation_id=cid,
            role="user",
            content=content,
        )

        if self.on_message:
            self.on_message(user_msg)

        response = await executor.ainvoke(user_msg)

        if self.on_complete:
            self.on_complete(response)

        return response

    async def stream(
        self, content: str, *, conversation_id: str | None = None, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream a response token by token.

        GAP-31: Accepts optional conversation_id for concurrent conversations.
        After the generator is exhausted, call ``last_stream_result`` to get
        the full ``Message`` with token metadata.
        """
        cid = conversation_id or self._default_conversation_id
        executor = self._get_or_create_executor(cid)
        user_msg = Message(
            conversation_id=cid,
            role="user",
            content=content,
        )
        async for chunk in executor.astream(user_msg):
            yield chunk

    @property
    def last_stream_result(self) -> Message | None:
        """Return the Message from the most recent ``stream()`` call."""
        executor = self._executors.get(self._default_conversation_id)
        if executor:
            return getattr(executor, "_last_stream_result", None)
        return None

    def reset_conversation(self, conversation_id: str | None = None) -> None:
        """Clear conversation history and start fresh.

        GAP-31: Accepts optional conversation_id to reset a specific conversation.
        """
        cid = conversation_id or self._default_conversation_id
        executor = self._executors.get(cid)
        if executor is None:
            return

        old_conversation_id = cid

        # GAP-79: Persist provenance before clearing — audit trail survives reset
        if self.provider and hasattr(self.provider, "save_provenance"):
            try:
                self.provider.save_provenance(executor.provenance_log)
            except Exception:
                pass
        elif self._provider and hasattr(self._provider, "save_provenance"):
            try:
                self._provider.save_provenance(executor.provenance_log)
            except Exception:
                pass

        executor.conversation_history.clear()
        executor.usage_records.clear()
        # NOTE: provenance_log is intentionally NOT cleared — provenance
        # persists independently of conversation history per plan.

        # Reset the per-conversation provider (not the agent's global provider)
        conv_provider = getattr(executor, "provider", None)
        if conv_provider and hasattr(conv_provider, "reset_history"):
            conv_provider.reset_history()

        if cid == self._default_conversation_id:
            new_id = uuid4().hex
            self._conversation_id = new_id
            self._default_conversation_id = new_id
            del self._executors[cid]
            self._executors[new_id] = executor
        else:
            del self._executors[cid]

        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentConversationResetEvent

            crewai_event_bus.emit(
                self,
                NewAgentConversationResetEvent(
                    conversation_id=old_conversation_id,
                    new_agent_id=self.id,
                ),
            )
        except Exception:
            pass

    def explain(self, conversation_id: str | None = None) -> list[ProvenanceEntry]:
        """Return the decision trace for this agent.

        GAP-31: Accepts optional conversation_id for a specific conversation.
        """
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentExplainRequestedEvent

            crewai_event_bus.emit(
                self,
                NewAgentExplainRequestedEvent(new_agent_id=self.id),
            )
        except Exception:
            pass

        cid = conversation_id or self._default_conversation_id
        executor = self._executors.get(cid)
        if executor is None:
            return []

        entries = list(executor.provenance_log)

        # GAP-88: Decouple from planning engine. Use a direct sync LLM call
        # for reasoning reconstruction — works in both sync and async contexts.
        needs_reasoning = any(not e.reasoning for e in entries)
        if needs_reasoning and self._llm_instance:
            try:
                from crewai.utilities.agent_utils import (
                    format_message_for_llm,
                    get_llm_response,
                )
                from crewai.utilities.types import LLMMessage

                log_text = "\n".join(
                    f"Step {i + 1}: {e.action} - inputs={e.inputs}, outcome={e.outcome}"
                    for i, e in enumerate(entries)
                )
                prompt = (
                    f"Given this execution trace, explain the reasoning behind each step:\n\n"
                    f"{log_text}\n\n"
                    f"For each step, provide a brief explanation of WHY the agent chose that action."
                )
                messages: list[LLMMessage] = [
                    format_message_for_llm(prompt, role="user")
                ]
                reasoning_text = get_llm_response(
                    llm=self._llm_instance,
                    messages=messages,
                    callbacks=[],
                )
                if reasoning_text:
                    reasoning_str = str(reasoning_text).strip()
                    for entry in entries:
                        if not entry.reasoning:
                            entry.reasoning = reasoning_str
            except Exception:
                pass

        return entries

    @property
    def memory_view(self) -> Any:
        """GAP-111: Read-only view of the agent's memory backend.

        Returns the underlying memory instance (supports .recall(), .save(), etc.)
        or None if memory is disabled. For a higher-level query API, use query_memory().
        """
        return self._memory_instance

    def query_memory(self, query: str, limit: int = 10) -> list[Any]:
        """Query the agent's memory for relevant information.

        GAP-45: Applies MemoryScope namespace and MemorySlice filters
        when configured.
        """
        if self._memory_instance is None:
            return []
        try:
            scoped_query = query
            if self._memory_namespace:
                scoped_query = f"[{self._memory_namespace}] {query}"

            results = self._memory_instance.recall(scoped_query, limit=limit)
            if not results:
                return []

            if self._memory_filter is not None:
                filtered = []
                for r in results:
                    r_str = str(r).lower() if r else ""
                    if (
                        self._memory_filter.user_id
                        and self._memory_filter.user_id.lower() not in r_str
                    ):
                        continue
                    filtered.append(r)
                return filtered

            return results or []
        except Exception:
            return []

    def get_conversation_history(self, conversation_id: str) -> list[Message]:
        """GAP-31: Get conversation history for a specific conversation."""
        executor = self._executors.get(conversation_id)
        if executor is None:
            return []
        return executor.conversation_history

    @property
    def conversation_history(self) -> list[Message]:
        """Return the default conversation's history."""
        executor = self._executors.get(self._default_conversation_id)
        if executor is None:
            return []
        return executor.conversation_history

    @property
    def last_prompt_stack(self) -> PromptStack | None:
        executor = self._executors.get(self._default_conversation_id)
        if executor is None:
            return None
        return executor.prompt_stack

    @property
    def usage_metrics(self) -> dict[str, int]:
        executor = self._executors.get(self._default_conversation_id)
        if executor is None:
            return {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tokens": 0,
                "total_actions": 0,
            }
        total_in = sum(r.input_tokens for r in executor.usage_records)
        total_out = sum(r.output_tokens for r in executor.usage_records)
        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
            "total_actions": len(executor.usage_records),
        }

    # ── GAP-40: Training → Canonical Memories ──────────────────

    def train(self, feedback: str, task_context: str = "") -> None:
        """Process training feedback as canonical memories.

        GAP-40: Instead of prompt-tuning, saves feedback as high-priority
        memories for the agent to recall during future conversations.
        """
        if not self._memory_instance:
            return

        canonical = f"Training feedback: {feedback}"
        if task_context:
            canonical = f"Context: {task_context}\nFeedback: {feedback}"

        try:
            self._memory_instance.remember(
                canonical,
                agent_role=self.role,
                importance=0.95,
            )
        except Exception:
            pass

        if self._dreaming_engine:
            try:
                self._dreaming_engine.add_training_feedback(feedback, task_context)
            except Exception:
                pass

    # ── GAP-24: Anaphora Resolution in Memory Encoding ─────────

    def prepare_memory_context(self, raw_text: str) -> str:
        """Prepare text for memory storage by resolving anaphora.

        GAP-24: Returns an enhanced prompt that the executor can use
        to resolve pronouns before saving to memory.
        """
        last_messages = (
            self.conversation_history[-5:] if self.conversation_history else []
        )
        context = "\n".join(f"{m.role}: {m.content}" for m in last_messages)
        return (
            f"Given this conversation context:\n{context}\n\n"
            f"Resolve all pronouns and references in the following text to their "
            f"full names/concepts. Only output the resolved text, nothing else:\n"
            f"{raw_text}"
        )

    def _resolve_anaphora(self, text: str, context: list[Message]) -> str:
        """Resolve pronouns in text using conversation context.

        GAP-24: Only triggers if the text contains pronouns.
        Requires an LLM call via the agent's LLM.
        """
        if not _ANAPHORA_PRONOUNS.search(text):
            return text

        llm = self._llm_instance
        if llm is None:
            return text

        context_str = "\n".join(f"{m.role}: {m.content}" for m in context[-5:])
        prompt = (
            f"Given this conversation context:\n{context_str}\n\n"
            f"Resolve all pronouns and references in the following text to their "
            f"full names/concepts. Only output the resolved text, nothing else:\n"
            f"{text}"
        )

        try:
            from crewai.utilities.agent_utils import (
                format_message_for_llm,
                get_llm_response,
            )
            from crewai.utilities.types import LLMMessage

            messages: list[LLMMessage] = [format_message_for_llm(prompt, role="user")]
            result = get_llm_response(
                llm=llm,
                messages=messages,
                callbacks=[],
            )
            resolved = str(result).strip()
            return resolved if resolved else text
        except Exception:
            return text
