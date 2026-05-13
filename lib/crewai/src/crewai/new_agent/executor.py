"""ConversationalAgentExecutor — message-based executor for NewAgent."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
import contextvars
import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

from crewai.new_agent.models import (
    AgentStatus,
    Artifact,
    Message,
    PromptStack,
    ProvenanceEntry,
    TokenUsage,
)
from crewai.utilities.agent_utils import (
    aget_llm_response,
    convert_tools_to_openai_schema,
    format_message_for_llm,
    handle_context_length,
    has_reached_max_iterations,
    is_context_length_exceeded,
    summarize_messages,
)
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.new_agent.new_agent import NewAgent

logger = logging.getLogger(__name__)

_current_conversation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "new_agent_conversation_id", default=""
)
_current_agent_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "new_agent_id", default=""
)


def get_current_conversation_id() -> str:
    return _current_conversation_id.get()


def get_current_agent_id() -> str:
    return _current_agent_id.get()


def _match_skill_trigger(text: str, phrase: str) -> bool:
    """Check if *phrase* appears in *text* as a coherent unit.

    Uses word-boundary matching to avoid false positives like
    "always use" matching the trigger "always do".
    """
    return bool(re.search(rf"\b{re.escape(phrase)}\b", text))


class ConversationalAgentExecutor(BaseModel):
    """Executor for NewAgent. Handles conversational turns with
    memory, provenance, and coworker delegation."""

    model_config = {"arbitrary_types_allowed": True}

    agent: Any = Field(default=None, exclude=True)
    provider: Any = Field(default=None, exclude=True)

    conversation_history: list[Message] = Field(default_factory=list)
    provenance_log: list[ProvenanceEntry] = Field(default_factory=list)
    prompt_stack: PromptStack | None = None
    usage_records: list[TokenUsage] = Field(default_factory=list)

    max_iter: int = 25
    verbose: bool = False

    _turn_start_time: float = PrivateAttr(default=0.0)
    _turn_input_tokens: int = PrivateAttr(default=0)
    _turn_output_tokens: int = PrivateAttr(default=0)
    _tools_used_this_turn: list[str] = PrivateAttr(default_factory=list)
    _delegations_this_turn: list[str] = PrivateAttr(default_factory=list)
    _tool_name_mapping: dict[str, Any] = PrivateAttr(default_factory=dict)
    _llm_prompt_tokens_before: int = PrivateAttr(default=0)
    _llm_completion_tokens_before: int = PrivateAttr(default=0)
    _tool_cache: dict[str, str] = PrivateAttr(default_factory=dict)
    # GAP-49: Sub-action token tracking
    _sub_action_tokens: list[TokenUsage] = PrivateAttr(default_factory=list)
    # GAP-33: Last checkpoint data for programmatic access
    _last_checkpoint: dict[str, Any] = PrivateAttr(default_factory=dict)
    # GAP-67: Artifacts collected during tool execution
    _turn_artifacts: list[Artifact] = PrivateAttr(default_factory=list)
    _last_stream_result: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Load persisted conversation history and provenance from provider on startup."""
        if self.provider and hasattr(self.provider, "get_history"):
            saved = self.provider.get_history()
            if saved:
                self.conversation_history.extend(saved)
        # GAP-50: Load persisted provenance entries
        if self.provider and hasattr(self.provider, "load_provenance"):
            try:
                saved_provenance = self.provider.load_provenance()
                if saved_provenance:
                    self.provenance_log.extend(saved_provenance)
            except Exception:
                pass

    def _build_prompt_stack(self, user_content: str = "") -> PromptStack:
        """Assemble the PromptStack from agent attributes.

        GAP-66: Layer order follows the plan specification:
        Soul -> Tools -> Memory -> Knowledge -> Coworkers -> Temporal
        """
        stack = PromptStack()
        agent = self.agent

        # 1. Soul layer
        soul = (
            f"You are {agent.role}.\n"
            f"Your goal: {agent.goal}\n"
            f"Background: {agent.backstory}"
        )
        stack.add("soul", soul, source="agent.role/goal/backstory")

        # 2. Tools layer
        if agent._resolved_tools:
            tool_descs = []
            for t in agent._resolved_tools:
                desc = t.description
                if "Tool Description:" in desc:
                    desc = desc.split("Tool Description:")[-1].strip()
                tool_descs.append(f"- {t.name}: {desc}")
            stack.add(
                "tools",
                "You have access to the following tools:\n" + "\n".join(tool_descs),
                source="agent.tools",
            )

        # 3. Memory layer
        memory_context = self._recall_memory(user_content)
        if memory_context:
            stack.add("memory", memory_context, source="memory.recall")

        # 4. Knowledge layer
        knowledge_context = self._query_knowledge(user_content)
        if knowledge_context:
            stack.add("knowledge", knowledge_context, source="agent.knowledge")

        # 4.5 Skills layer
        skill_builder = getattr(agent, "_skill_builder", None)
        if skill_builder:
            parts = []
            skills_context = skill_builder.format_skills_context()
            if skills_context:
                parts.append(skills_context)
            if agent.settings.can_build_skills:
                parts.append(
                    "You can learn new skills from instructions the user gives you. "
                    "When the user asks you to remember a process, encode a workflow, "
                    "or create a skill, a skill suggestion will be generated automatically — "
                    "do NOT use file-writing tools to create skill files yourself."
                )
            if parts:
                stack.add("skills", "\n\n".join(parts), source="agent.skills")
        else:
            active_skills = getattr(agent, "_active_skills", [])
            if active_skills:
                try:
                    from crewai.skills.loader import format_skill_context

                    sections = [format_skill_context(s) for s in active_skills]
                    stack.add("skills", "\n\n".join(sections), source="agent.skills")
                except Exception:
                    pass

        # 5. Coworkers layer
        if agent._coworker_tools:
            cw_descs = []
            for t in agent._coworker_tools:
                desc = t.description
                if "Tool Description:" in desc:
                    desc = desc.split("Tool Description:")[-1].strip()
                cw_descs.append(f"- {t.name}: {desc}")
            stack.add(
                "coworkers",
                "You have coworkers with specialized expertise. "
                "When a request involves work outside your core specialty, "
                "delegate to the appropriate coworker using their tool. "
                "Delegation is preferred over attempting work you're not specialized in.\n\n"
                "Available coworkers:\n" + "\n".join(cw_descs),
                source="agent.coworkers",
            )

        # 6. Temporal layer
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        stack.add("temporal", f"Current date and time: {now}", source="system")

        return stack

    def _recall_memory(self, query: str) -> str:
        """Recall relevant memories for the current query.

        GAP-120: Filters out memories whose scope doesn't match the current
        conversation context, preventing user-specific preferences from leaking
        into other users' system prompts.
        """
        agent = self.agent
        if not agent.settings.memory_enabled:
            return ""
        memory = getattr(agent, "_memory_instance", None)
        if memory is None:
            return ""
        try:
            matches = memory.recall(query, limit=5)
            if not matches:
                return ""

            conv_id = (
                self.conversation_history[0].conversation_id
                if self.conversation_history
                else ""
            )
            scope = self._get_provider_scope()
            provider_user = scope.get("user_id", "")
            provider_channel = scope.get("channel_id", "")
            provider_team = scope.get("team_id", "")
            if (
                not provider_user
                and self.provider
                and hasattr(self.provider, "user_id")
            ):
                provider_user = getattr(self.provider, "user_id", "") or ""

            filtered: list[Any] = []
            for m in matches:
                meta = getattr(m, "metadata", None) or {}
                if isinstance(meta, str):
                    meta = {}
                if meta.get("type") == "provenance":
                    continue
                mem_conv = meta.get("conversation_id", "")
                if mem_conv and conv_id and mem_conv != conv_id:
                    continue
                mem_user = meta.get("user_id", "")
                if mem_user and provider_user and mem_user != provider_user:
                    continue
                mem_channel = meta.get("channel_id", "")
                if mem_channel and provider_channel and mem_channel != provider_channel:
                    continue
                mem_team = meta.get("team_id", "")
                if mem_team and provider_team and mem_team != provider_team:
                    continue
                filtered.append(m)

            if not filtered:
                return ""

            lines = ["Relevant memories:"]
            for m in filtered:
                content = getattr(m, "content", None) or getattr(
                    getattr(m, "record", None), "content", ""
                )
                if content:
                    lines.append(f"- {content}")
            try:
                from crewai.new_agent.events import NewAgentMemoryRecallEvent

                self._emit_event(
                    NewAgentMemoryRecallEvent(
                        new_agent_id=str(self.agent.id),
                        results_count=len(filtered),
                    )
                )
            except Exception:
                pass
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Memory recall failed: {e}")
            return ""

    def _query_knowledge(self, query: str) -> str:
        """Query agent knowledge sources for relevant context."""
        agent = self.agent
        knowledge = getattr(agent, "knowledge", None)
        if knowledge is None and not getattr(agent, "knowledge_sources", []):
            return ""
        try:
            if knowledge is not None and hasattr(knowledge, "query"):
                results = knowledge.query(query)
                if results:
                    lines = ["Relevant knowledge:"]
                    for r in results[:5]:
                        content = getattr(r, "content", str(r))
                        if content:
                            lines.append(f"- {content}")
                    knowledge_text = "\n".join(lines)
                    if len(lines) > 1:
                        try:
                            from crewai.new_agent.events import (
                                NewAgentKnowledgeQueryEvent,
                            )

                            self._emit_event(
                                NewAgentKnowledgeQueryEvent(
                                    new_agent_id=str(self.agent.id),
                                )
                            )
                        except Exception:
                            pass
                        return knowledge_text
        except Exception as e:
            logger.debug(f"Knowledge query failed: {e}")
        return ""

    def _build_llm_messages(self, user_message: Message) -> list[LLMMessage]:
        """Convert conversation history + prompt stack into LLM messages."""
        messages: list[LLMMessage] = []

        system_prompt = self.prompt_stack.assemble() if self.prompt_stack else ""
        if system_prompt:
            messages.append(format_message_for_llm(system_prompt, role="system"))

        settings = self.agent.settings
        history = self.conversation_history
        if settings.max_history_messages is not None:
            history = history[-settings.max_history_messages :]

        for msg in history:
            if msg.role == "user":
                messages.append(format_message_for_llm(msg.content, role="user"))
            elif msg.role in ("agent", "coworker"):
                messages.append(format_message_for_llm(msg.content, role="assistant"))
            elif msg.role == "system":
                messages.append(format_message_for_llm(msg.content, role="system"))

        messages.append(format_message_for_llm(user_message.content, role="user"))
        return messages

    async def _emit_status(
        self, state: str, detail: str | None = None, **kwargs: Any
    ) -> None:
        """Emit a status update via the provider and event bus."""
        elapsed = int((time.monotonic() - self._turn_start_time) * 1000)
        status = AgentStatus(
            state=state,
            detail=detail,
            elapsed_ms=elapsed,
            input_tokens=self._turn_input_tokens,
            output_tokens=self._turn_output_tokens,
            **kwargs,
        )
        if self.provider is not None:
            try:
                await self.provider.send_status(status)
            except Exception:
                pass
        try:
            from crewai.new_agent.events import NewAgentStatusUpdateEvent

            self._emit_event(
                NewAgentStatusUpdateEvent(
                    state=state,
                    detail=detail,
                    input_tokens=status.input_tokens,
                    output_tokens=status.output_tokens,
                    elapsed_ms=status.elapsed_ms,
                    new_agent_id=getattr(self.agent, "id", ""),
                )
            )
        except Exception:
            pass

    def _track_tokens_from_llm(self) -> None:
        """Read token counts from the LLM's internal usage tracker."""
        llm = self.agent._llm_instance
        if llm is None:
            return
        usage = getattr(llm, "_token_usage", None)
        if usage is None:
            return
        current_prompt = usage.get("prompt_tokens", 0)
        current_completion = usage.get("completion_tokens", 0)
        self._turn_input_tokens += max(
            0, current_prompt - self._llm_prompt_tokens_before
        )
        self._turn_output_tokens += max(
            0, current_completion - self._llm_completion_tokens_before
        )
        self._llm_prompt_tokens_before = current_prompt
        self._llm_completion_tokens_before = current_completion

    def _resolve_function_calling_llm(self) -> Any:
        """Resolve a separate LLM for function calling if configured."""
        fc_ref = getattr(self.agent, "function_calling_llm", None)
        if fc_ref is None:
            return None
        if isinstance(fc_ref, str):
            from crewai.utilities.llm_utils import create_llm

            return create_llm(fc_ref)
        return fc_ref

    @staticmethod
    def _extract_reasoning_from_text(text: str) -> str:
        """GAP-121: Extract reasoning from the model's response text (no LLM call).

        Looks for common reasoning patterns in the response and extracts them.
        Used by the 'standard' provenance tier when extended thinking is off.
        """
        import re

        # Look for explicit reasoning markers
        patterns = [
            r"(?:My reasoning|My rationale|Here's why|The reason)\s*(?:is|:)\s*(.+?)(?:\n\n|\Z)",
            r"(?:I (?:chose|decided|opted|selected|recommend|suggest)(?:ed)?)\s+(?:to |this |that |because )(.+?)(?:\n\n|\.|$)",
            r"(?:Because|Since|Given that)\s+(.+?)(?:,\s*I|\.\s|\n)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                extracted = match.group(1).strip()
                if len(extracted) > 15:
                    return extracted[:300]
        # Fallback: use first sentence as a lightweight summary
        first_sentence = text.split(".")[0].strip() if text else ""
        if len(first_sentence) > 20:
            return first_sentence[:200]
        return ""

    async def _maybe_generate_reasoning(
        self, action: str, inputs: dict[str, Any], outcome: str
    ) -> str:
        """Generate explicit reasoning for provenance if provenance_detail is 'detailed'.

        Returns '' for 'minimal' detail level.
        For 'standard', extracts reasoning from outcome text (free, no LLM call).
        Makes an additional LLM call for 'detailed' level.
        """
        detail = self.agent.settings.provenance_detail
        if detail == "minimal":
            return ""
        # GAP-121: Standard tier extracts reasoning from model output (no LLM call)
        if detail == "standard":
            return self._extract_reasoning_from_text(outcome)

        # detailed: make an LLM call to generate reasoning
        llm = self.agent._llm_instance
        if llm is None:
            return ""

        prompt = (
            f"Briefly explain the reasoning behind this action.\n"
            f"Action: {action}\n"
            f"Inputs: {json.dumps(inputs, default=str)[:500]}\n"
            f"Outcome: {outcome[:500]}\n"
            f"Reasoning (1-2 sentences):"
        )
        messages: list[LLMMessage] = [
            format_message_for_llm(prompt, role="user"),
        ]
        callbacks: list[TokenCalcHandler] = [TokenCalcHandler()]
        try:
            from crewai.new_agent.events import (
                NewAgentLLMCallCompletedEvent,
                NewAgentLLMCallFailedEvent,
                NewAgentLLMCallStartedEvent,
            )

            llm_model = getattr(llm, "model", "") or ""
            self._emit_event(
                NewAgentLLMCallStartedEvent(
                    new_agent_id=str(self.agent.id),
                    model=llm_model,
                )
            )
            call_start = time.monotonic()
            answer = await aget_llm_response(
                llm=llm,
                messages=messages,
                callbacks=callbacks,
                printer=_NullPrinter(),
                verbose=False,
            )
            self._track_tokens_from_llm()
            call_elapsed = int((time.monotonic() - call_start) * 1000)
            self._emit_event(
                NewAgentLLMCallCompletedEvent(
                    new_agent_id=str(self.agent.id),
                    model=llm_model,
                    input_tokens=self._turn_input_tokens,
                    output_tokens=self._turn_output_tokens,
                    response_time_ms=call_elapsed,
                )
            )
            return str(answer).strip() if answer else ""
        except Exception as e:
            try:
                self._emit_event(
                    NewAgentLLMCallFailedEvent(
                        new_agent_id=str(self.agent.id),
                        error=str(e),
                    )
                )
            except Exception:
                pass
            logger.debug(f"Reasoning generation failed: {e}")
            return ""

    def _estimate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float | None:
        """Approximate cost in USD based on common model pricing per 1M tokens."""
        costs = {
            "gpt-4.1-nano": (0.10, 0.40),
            "gpt-4.1-mini": (0.40, 1.60),
            "gpt-4.1": (2.00, 8.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "gpt-4": (30.00, 60.00),
            "gpt-5-mini": (0.25, 2.00),
            "gpt-5.5": (5.00, 30.00),
            "gpt-5.4": (2.50, 15.00),
            "gpt-5": (1.25, 10.00),
            "o4-mini": (1.10, 4.40),
            "o3-mini": (1.10, 4.40),
            "o3": (2.00, 8.00),
            "claude-opus": (5.00, 25.00),
            "claude-sonnet": (3.00, 15.00),
            "claude-haiku": (1.00, 5.00),
        }
        for key, (inp_cost, out_cost) in costs.items():
            if key in model.lower():
                return (input_tokens * inp_cost + output_tokens * out_cost) / 1_000_000
        return None

    def _record_token_usage(self, action: str, model: str, **kwargs: Any) -> None:
        agent_id = str(self.agent.id) if self.agent else ""
        conv_id = (
            self.conversation_history[0].conversation_id
            if self.conversation_history
            else ""
        )
        self.usage_records.append(
            TokenUsage(
                action=action,
                agent_id=agent_id,
                conversation_id=conv_id,
                input_tokens=self._turn_input_tokens,
                output_tokens=self._turn_output_tokens,
                model=model,
                **kwargs,
            )
        )
        # GAP-118: Emit token usage event for platform billing
        try:
            from crewai.new_agent.events import NewAgentTokenUsageEvent

            self._emit_event(
                NewAgentTokenUsageEvent(
                    new_agent_id=agent_id,
                    conversation_id=conv_id,
                    action=action,
                    input_tokens=self._turn_input_tokens,
                    output_tokens=self._turn_output_tokens,
                    model=model,
                )
            )
        except Exception:
            pass

    def _record_sub_action_token_usage(
        self, action: str, model: str, input_tokens: int = 0, output_tokens: int = 0
    ) -> None:
        """GAP-49: Record token usage for a sub-action (planning, guardrail, reasoning, etc.)."""
        entry = TokenUsage(
            action=action,
            agent_id=str(self.agent.id) if self.agent else "",
            conversation_id=self.conversation_history[0].conversation_id
            if self.conversation_history
            else "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
        self._sub_action_tokens.append(entry)

    def _extract_thinking_output(self, answer: Any) -> str:
        """GAP-53: Extract thinking/reasoning output from the LLM response if available.

        Checks for thinking blocks in the raw LLM response. This is free reasoning
        that was already generated during the LLM call.
        """
        try:
            # Check for thinking attribute on the response object
            if hasattr(answer, "thinking"):
                return str(answer.thinking) if answer.thinking else ""

            # Check if the LLM instance has a cached thinking output
            llm = self.agent._llm_instance
            if llm is not None:
                # Some LLMs store thinking in _last_thinking or similar
                thinking = getattr(llm, "_last_thinking", None)
                if thinking:
                    return str(thinking)
                # Check litellm-style response metadata
                last_response = getattr(llm, "_last_response", None)
                if last_response:
                    choices = getattr(last_response, "choices", None)
                    if choices:
                        msg = getattr(choices[0], "message", None)
                        if msg:
                            thinking = getattr(msg, "thinking", None) or getattr(
                                msg, "reasoning_content", None
                            )
                            if thinking:
                                return str(thinking)
        except Exception:
            pass
        return ""

    def _detect_artifacts(self, tool_name: str, result_str: str) -> list[Artifact]:
        """GAP-67: Detect artifacts from tool results.

        Heuristics:
        - File paths that exist -> Artifact(type="file")
        - Valid JSON -> Artifact(type="json")
        - URLs -> Artifact(type="url")
        - Very long output (> 2000 chars) -> Artifact(type="code")
        """
        import os

        artifacts: list[Artifact] = []
        if not result_str:
            return artifacts

        # Check for URL patterns
        url_pattern = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+')
        urls = url_pattern.findall(result_str)
        for url in urls:
            artifacts.append(
                Artifact(
                    type="url",
                    name=f"{tool_name}_url",
                    content=url,
                )
            )

        # Check for file paths (lines that look like existing file paths)
        for line in result_str.split("\n"):
            line = line.strip()
            if line and os.path.sep in line:
                # Try to extract a path-like string
                potential_path = line.strip("'\"` ")
                try:
                    if os.path.exists(potential_path) and os.path.isfile(
                        potential_path
                    ):
                        artifacts.append(
                            Artifact(
                                type="file",
                                name=os.path.basename(potential_path),
                                content=potential_path,
                            )
                        )
                except (OSError, ValueError):
                    pass

        # Check for valid JSON (if the whole result is JSON)
        if not artifacts or len(artifacts) == len(urls):
            stripped = result_str.strip()
            if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
                try:
                    json.loads(stripped)
                    artifacts.append(
                        Artifact(
                            type="json",
                            name=f"{tool_name}_result",
                            content=stripped,
                        )
                    )
                except (json.JSONDecodeError, ValueError):
                    pass

        # Very long output heuristic
        if not artifacts and len(result_str) > 2000:
            artifacts.append(
                Artifact(
                    type="code",
                    name=f"{tool_name}_output",
                    content=result_str,
                )
            )

        return artifacts

    def _emit_checkpoint(self) -> None:
        """GAP-33: Emit checkpoint data after a turn completes.

        Serializes current state for recovery/inspection. If the checkpoint
        event infrastructure is not available, stores data on self._last_checkpoint.
        """
        try:
            checkpoint_data: dict[str, Any] = {
                "prompt_stack": [
                    layer.model_dump(mode="json")
                    for layer in (self.prompt_stack.layers if self.prompt_stack else [])
                ],
                "provenance_log": [
                    entry.model_dump(mode="json") for entry in self.provenance_log
                ],
                "conversation_history": [
                    msg.model_dump(mode="json") for msg in self.conversation_history
                ],
                "token_usage": {
                    "input_tokens": self._turn_input_tokens,
                    "output_tokens": self._turn_output_tokens,
                },
                "conversation_id": (
                    self.conversation_history[0].conversation_id
                    if self.conversation_history
                    else ""
                ),
            }
            self._last_checkpoint = checkpoint_data

            # Try to emit as an event
            try:
                from crewai.events.event_bus import crewai_event_bus
                from crewai.utilities.events.checkpoint_events import CheckpointEvent

                crewai_event_bus.emit(self, CheckpointEvent(data=checkpoint_data))
            except (ImportError, Exception):
                pass
        except Exception:
            pass

    def get_checkpoint(self) -> dict[str, Any]:
        """Return the last checkpoint data for external state capture (e.g., Flow persistence)."""
        return dict(self._last_checkpoint)

    def invoke(self, user_message: Message) -> Message:
        """Process a single conversational turn (sync)."""
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.ainvoke(user_message))
                return future.result()
        else:
            return asyncio.run(self.ainvoke(user_message))

    def _maybe_summarize_history(self) -> None:
        """Proactively cap conversation_history to prevent unbounded growth.

        Hard cap only — LLM-based summarization happens in
        _proactive_summarize_messages() after llm_messages are built.
        """
        hard_cap = 500
        if len(self.conversation_history) > hard_cap:
            self.conversation_history = self.conversation_history[-hard_cap:]

    def _proactive_summarize_messages(
        self, llm_messages: list[LLMMessage], callbacks: list[Any]
    ) -> None:
        """Summarize llm_messages in-place if they approach the context window.

        Reuses the existing summarize_messages() from agent_utils which handles
        chunking, parallel summarization, and file attachment preservation.
        """
        if not self.agent.settings.respect_context_window:
            return

        llm = self.agent._llm_instance
        if llm is None:
            return

        ctx_size = llm.get_context_window_size()
        total_chars = sum(len(str(m.get("content", ""))) for m in llm_messages)
        est_tokens = total_chars // 3

        if est_tokens < int(ctx_size * 0.60):
            return

        try:
            summarize_messages(
                messages=llm_messages,
                llm=llm,
                callbacks=callbacks,
                verbose=self.verbose,
            )
            self._emit_event_context_summarized()
        except Exception as e:
            logger.debug(f"Proactive summarization failed: {e}")

    def _emit_event_context_summarized(self) -> None:
        try:
            from crewai.new_agent.events import NewAgentContextSummarizedEvent

            self._emit_event(
                NewAgentContextSummarizedEvent(
                    new_agent_id=str(self.agent.id),
                )
            )
        except Exception:
            pass

    def _emit_event(self, event: Any) -> None:
        """Emit an event on the CrewAI event bus."""
        try:
            from crewai.events.event_bus import crewai_event_bus

            crewai_event_bus.emit(self, event)
        except Exception:
            pass

    async def _run_guardrail(self, response_text: str) -> str:
        """Run the agent's guardrail on the response. Returns (possibly modified) text."""
        guardrail = self.agent.guardrail
        if guardrail is None:
            return response_text

        from crewai.tasks.llm_guardrail import LLMGuardrail

        max_retries = self.agent.settings.max_retry_limit
        for attempt in range(max_retries + 1):
            try:
                if isinstance(guardrail, LLMGuardrail):
                    passed, feedback = await self._run_llm_guardrail(
                        response_text, guardrail
                    )
                    if passed:
                        self._emit_event_guardrail("llm", True, attempt)
                        return response_text
                    self._emit_event_guardrail("llm", False, attempt)
                    if attempt < max_retries:
                        response_text = await self._regenerate_with_feedback(
                            response_text, str(feedback)
                        )
                        continue
                    return response_text
                if callable(guardrail) and not isinstance(guardrail, str):
                    result = guardrail(response_text)
                    if isinstance(result, tuple):
                        passed, feedback = result
                    elif isinstance(result, bool):
                        passed, feedback = result, ""
                    else:
                        passed, feedback = bool(result), ""

                    if passed:
                        self._emit_event_guardrail("code", True, attempt)
                        return response_text
                    self._emit_event_guardrail("code", False, attempt)
                    if attempt < max_retries:
                        response_text = await self._regenerate_with_feedback(
                            response_text, str(feedback)
                        )
                        continue
                    return response_text
                if isinstance(guardrail, str):
                    return response_text
                return response_text
            except Exception as e:
                logger.warning(f"Guardrail error: {e}")
                return response_text
        return response_text

    async def _run_llm_guardrail(
        self, response_text: str, guardrail: Any
    ) -> tuple[bool, str]:
        """Evaluate response against an LLM-based guardrail.

        Returns:
            A tuple of (passed, feedback). ``passed`` is True when the
            response satisfies the guardrail instructions.
        """
        llm = getattr(guardrail, "llm", None) or self.agent._llm_instance
        if llm is None:
            return True, ""

        # If the guardrail stores the LLM as a string, resolve it.
        if isinstance(llm, str):
            from crewai.utilities.llm_utils import create_llm

            llm = create_llm(llm)

        instructions = getattr(guardrail, "description", "") or ""
        prompt = (
            "Does this response violate any of these rules? "
            f"Rules: {instructions}. "
            f"Response: {response_text}. "
            "Answer with PASS or FAIL: reason"
        )

        messages: list[LLMMessage] = [
            format_message_for_llm(prompt, role="user"),
        ]
        callbacks: list[TokenCalcHandler] = [TokenCalcHandler()]
        guardrail_model = getattr(llm, "model", "") or ""
        try:
            # GAP-03: Emit LLM call started event for guardrail
            try:
                from crewai.new_agent.events import NewAgentLLMCallStartedEvent

                self._emit_event(
                    NewAgentLLMCallStartedEvent(
                        new_agent_id=str(self.agent.id),
                        model=guardrail_model,
                    )
                )
            except Exception:
                pass

            # GAP-49: Track tokens before guardrail
            _guardrail_in_before = self._turn_input_tokens
            _guardrail_out_before = self._turn_output_tokens
            guardrail_call_start = time.monotonic()
            answer = await aget_llm_response(
                llm=llm,
                messages=messages,
                callbacks=callbacks,
                printer=_NullPrinter(),
                verbose=False,
            )
            self._track_tokens_from_llm()
            guardrail_call_elapsed = int(
                (time.monotonic() - guardrail_call_start) * 1000
            )

            # GAP-49: Record sub-action tokens for guardrail
            _gr_in = self._turn_input_tokens - _guardrail_in_before
            _gr_out = self._turn_output_tokens - _guardrail_out_before
            if _gr_in > 0 or _gr_out > 0:
                self._record_sub_action_token_usage(
                    "guardrail", guardrail_model, _gr_in, _gr_out
                )

            # GAP-03: Emit LLM call completed event for guardrail
            try:
                from crewai.new_agent.events import NewAgentLLMCallCompletedEvent

                self._emit_event(
                    NewAgentLLMCallCompletedEvent(
                        new_agent_id=str(self.agent.id),
                        model=guardrail_model,
                        input_tokens=self._turn_input_tokens,
                        output_tokens=self._turn_output_tokens,
                        response_time_ms=guardrail_call_elapsed,
                    )
                )
            except Exception:
                pass

            answer_str = str(answer).strip()

            if answer_str.upper().startswith("PASS"):
                return True, ""
            if answer_str.upper().startswith("FAIL"):
                # Extract feedback after "FAIL:" or "FAIL "
                feedback = answer_str
                for prefix in ("FAIL:", "FAIL"):
                    if feedback.upper().startswith(prefix):
                        feedback = feedback[len(prefix) :].strip()
                        break
                return False, feedback
            # Ambiguous answer — treat as pass to avoid spurious retries
            return True, ""
        except Exception as e:
            # GAP-03: Emit LLM call failed event for guardrail
            try:
                from crewai.new_agent.events import NewAgentLLMCallFailedEvent

                self._emit_event(
                    NewAgentLLMCallFailedEvent(
                        new_agent_id=str(self.agent.id),
                        error=str(e),
                    )
                )
            except Exception:
                pass
            logger.warning(f"LLM guardrail evaluation failed: {e}")
            return True, ""

    def _emit_event_guardrail(
        self, guardrail_type: str, passed: bool, retries: int
    ) -> None:
        try:
            from crewai.new_agent.events import (
                NewAgentGuardrailPassedEvent,
                NewAgentGuardrailRejectedEvent,
            )

            if passed:
                self._emit_event(
                    NewAgentGuardrailPassedEvent(
                        new_agent_id=str(self.agent.id),
                        guardrail_type=guardrail_type,
                    )
                )
            else:
                self._emit_event(
                    NewAgentGuardrailRejectedEvent(
                        new_agent_id=str(self.agent.id),
                        guardrail_type=guardrail_type,
                        retries=retries,
                    )
                )
        except Exception:
            pass

    async def _regenerate_with_feedback(self, original: str, feedback: str) -> str:
        """Ask the LLM to regenerate the response incorporating guardrail feedback."""
        llm = self.agent._llm_instance
        if llm is None:
            return original

        messages: list[LLMMessage] = [
            format_message_for_llm(
                f"Your previous response was rejected by a guardrail.\n"
                f"Feedback: {feedback}\n\n"
                f"Your original response:\n{original}\n\n"
                f"Please regenerate your response addressing the feedback.",
                role="user",
            )
        ]
        callbacks: list[TokenCalcHandler] = [TokenCalcHandler()]
        regen_model = getattr(llm, "model", "") or ""
        try:
            # GAP-03: Emit LLM call started event for regeneration
            try:
                from crewai.new_agent.events import NewAgentLLMCallStartedEvent

                self._emit_event(
                    NewAgentLLMCallStartedEvent(
                        new_agent_id=str(self.agent.id),
                        model=regen_model,
                    )
                )
            except Exception:
                pass

            regen_call_start = time.monotonic()
            answer = await aget_llm_response(
                llm=llm,
                messages=messages,
                callbacks=callbacks,
                printer=_NullPrinter(),
                verbose=False,
            )
            self._track_tokens_from_llm()
            regen_call_elapsed = int((time.monotonic() - regen_call_start) * 1000)

            # GAP-03: Emit LLM call completed event for regeneration
            try:
                from crewai.new_agent.events import NewAgentLLMCallCompletedEvent

                self._emit_event(
                    NewAgentLLMCallCompletedEvent(
                        new_agent_id=str(self.agent.id),
                        model=regen_model,
                        input_tokens=self._turn_input_tokens,
                        output_tokens=self._turn_output_tokens,
                        response_time_ms=regen_call_elapsed,
                    )
                )
            except Exception:
                pass

            return str(answer) if answer else original
        except Exception as e:
            # GAP-03: Emit LLM call failed event for regeneration
            try:
                from crewai.new_agent.events import NewAgentLLMCallFailedEvent

                self._emit_event(
                    NewAgentLLMCallFailedEvent(
                        new_agent_id=str(self.agent.id),
                        error=str(e),
                    )
                )
            except Exception:
                pass
            return original

    async def _parse_structured_output(self, text: str) -> BaseModel | None:
        """Parse the response text into the agent's response_model.

        Strategy:
        1. Try to parse ``text`` as JSON directly into the response_model.
        2. If that fails, ask the LLM to extract structured data matching
           the model's JSON schema.

        Returns the parsed Pydantic object, or ``None`` on failure.
        """
        response_model: type[BaseModel] = self.agent.response_model

        # 1. Attempt direct JSON parse
        try:
            return response_model.model_validate_json(text)
        except Exception:
            pass

        # Also try parsing after stripping markdown code fences
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            # Remove first and last lines (``` markers)
            inner = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            try:
                return response_model.model_validate_json(inner)
            except Exception:
                pass

        # 2. Fall back to LLM extraction
        llm = self.agent._llm_instance
        if llm is None:
            return None

        schema_json = json.dumps(response_model.model_json_schema(), indent=2)
        extraction_prompt = (
            "Extract structured data from the following text and return "
            "ONLY valid JSON matching this schema. Do not include any "
            "explanation or markdown formatting — output raw JSON only.\n\n"
            f"JSON Schema:\n{schema_json}\n\n"
            f"Text:\n{text}"
        )

        messages: list[LLMMessage] = [
            format_message_for_llm(extraction_prompt, role="user"),
        ]
        callbacks: list[TokenCalcHandler] = [TokenCalcHandler()]
        extract_model = getattr(llm, "model", "") or ""
        try:
            # GAP-03: Emit LLM call started event for structured extraction
            try:
                from crewai.new_agent.events import NewAgentLLMCallStartedEvent

                self._emit_event(
                    NewAgentLLMCallStartedEvent(
                        new_agent_id=str(self.agent.id),
                        model=extract_model,
                    )
                )
            except Exception:
                pass

            extract_call_start = time.monotonic()
            answer = await aget_llm_response(
                llm=llm,
                messages=messages,
                callbacks=callbacks,
                printer=_NullPrinter(),
                verbose=False,
            )
            self._track_tokens_from_llm()
            extract_call_elapsed = int((time.monotonic() - extract_call_start) * 1000)

            # GAP-03: Emit LLM call completed event for structured extraction
            try:
                from crewai.new_agent.events import NewAgentLLMCallCompletedEvent

                self._emit_event(
                    NewAgentLLMCallCompletedEvent(
                        new_agent_id=str(self.agent.id),
                        model=extract_model,
                        input_tokens=self._turn_input_tokens,
                        output_tokens=self._turn_output_tokens,
                        response_time_ms=extract_call_elapsed,
                    )
                )
            except Exception:
                pass

            answer_str = str(answer).strip()

            # Strip markdown code fences if present
            if answer_str.startswith("```"):
                lines = answer_str.split("\n")
                answer_str = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )

            return response_model.model_validate_json(answer_str)
        except Exception as e:
            # GAP-03: Emit LLM call failed event for structured extraction
            try:
                from crewai.new_agent.events import NewAgentLLMCallFailedEvent

                self._emit_event(
                    NewAgentLLMCallFailedEvent(
                        new_agent_id=str(self.agent.id),
                        error=str(e),
                    )
                )
            except Exception:
                pass
            logger.debug(f"Structured output parsing failed: {e}")
            return None

    def _get_provider_scope(self) -> dict[str, str]:
        """Get scope context from the provider for multi-tenant isolation."""
        provider = getattr(self.agent, "_provider", None)
        if provider and hasattr(provider, "get_scope"):
            try:
                result = provider.get_scope()
                if isinstance(result, dict):
                    return {
                        k: v
                        for k, v in result.items()
                        if isinstance(k, str) and isinstance(v, str)
                    }
            except Exception:
                pass
        return {}

    def _persist_provenance_to_memory(self, entry: ProvenanceEntry) -> None:
        """Save provenance entry to memory backend for long-term auditing."""
        if not self.agent._memory_instance:
            return
        try:
            value = f"[provenance] {entry.action}: {entry.outcome or ''}"
            metadata: dict[str, Any] = {
                "type": "provenance",
                "action": entry.action,
                "conversation_id": entry.conversation_id,
            }
            metadata.update(self._get_provider_scope())
            self.agent._memory_instance.remember(
                value=value,
                metadata=metadata,
            )
        except Exception:
            pass

    def _save_to_memory(self, user_message: Message, agent_message: Message) -> None:
        """Save conversation turn to memory for future recall."""
        agent = self.agent
        if not agent.settings.memory_enabled:
            return
        memory = getattr(agent, "_memory_instance", None)
        if memory is None:
            return
        try:
            raw = (
                f"User asked: {user_message.content}\n"
                f"Agent ({agent.role}) responded: {agent_message.content}"
            )
            # GAP-24: Anaphora resolution before memory encoding
            if hasattr(agent, "_resolve_anaphora") and callable(
                agent._resolve_anaphora
            ):
                try:
                    resolved = agent._resolve_anaphora(raw, self.conversation_history)
                    if resolved and resolved != raw:
                        raw = resolved
                except Exception:
                    pass
            extracted = memory.extract_memories(raw)
            if extracted:
                memory.remember_many(extracted, agent_role=agent.role)
                try:
                    from crewai.new_agent.events import NewAgentMemorySaveEvent

                    self._emit_event(
                        NewAgentMemorySaveEvent(
                            new_agent_id=str(self.agent.id),
                        )
                    )
                except Exception:
                    pass
                dreaming = getattr(agent, "_dreaming_engine", None)
                if dreaming:
                    dreaming.increment_memory_count()
        except Exception as e:
            logger.debug(f"Memory save failed: {e}")

    async def ainvoke(self, user_message: Message) -> Message:
        """Process a single conversational turn (async)."""
        _current_conversation_id.set(user_message.conversation_id or "")
        _current_agent_id.set(str(getattr(self.agent, "id", "")))

        self._turn_start_time = time.monotonic()
        self._turn_input_tokens = 0
        self._turn_output_tokens = 0
        self._tools_used_this_turn = []
        self._delegations_this_turn = []
        self._tool_cache = {}
        self._sub_action_tokens = []
        self._turn_artifacts = []

        # GAP-97: Proactively trim conversation history before building prompt
        self._maybe_summarize_history()

        # GAP-46: Telemetry execution_started span
        _telemetry_span = None
        try:
            if hasattr(self.agent, "_telemetry") and self.agent._telemetry:
                _telemetry_span = self.agent._telemetry.execution_started(
                    agent_id=str(self.agent.id),
                    conversation_id=getattr(self.agent, "_conversation_id", ""),
                    model=str(
                        getattr(self.agent._llm_instance, "model", "unknown")
                        if self.agent._llm_instance
                        else "unknown"
                    ),
                )
        except Exception:
            pass

        # GAP-32: max_execution_time enforcement
        max_time = getattr(self.agent, "max_execution_time", None)
        deadline = (time.monotonic() + max_time) if max_time else None

        llm = self.agent._llm_instance
        if llm is not None:
            usage = getattr(llm, "_token_usage", None) or {}
            self._llm_prompt_tokens_before = usage.get("prompt_tokens", 0)
            self._llm_completion_tokens_before = usage.get("completion_tokens", 0)

        self._emit_event_message_received(user_message)

        await self._emit_status("recalling", "Searching memory for relevant context")
        self.prompt_stack = self._build_prompt_stack(user_content=user_message.content)

        # Handle pending suggestion responses before new detection
        conv_id = (
            self.conversation_history[0].conversation_id
            if self.conversation_history
            else ""
        )
        skill_builder = getattr(self.agent, "_skill_builder", None)
        kd = getattr(self.agent, "_knowledge_discovery", None)

        if skill_builder and skill_builder.pending_suggestions:
            result = skill_builder.handle_suggestion_response(user_message.content)
            if result and result.get("action") in ("confirmed", "rejected"):
                from crewai.new_agent.models import Message as AgentMessage

                if result["action"] == "confirmed":
                    active = skill_builder.get_active_skills()
                    skills_list = "\n".join(
                        f"  **{s.name}** — {getattr(s, 'description', '')}"
                        for s in active
                    )
                    reply = (
                        f"Skill **{result['name']}** saved and activated.\n\n"
                        f"Active Skills ({len(active)}):\n{skills_list}"
                    )
                else:
                    reply = f"Skill suggestion **{result['name']}** dismissed."
                reply_msg = AgentMessage(
                    role="agent",
                    content=reply,
                    sender=self.agent.role,
                    conversation_id=conv_id,
                )
                self.conversation_history.append(user_message)
                self.conversation_history.append(reply_msg)
                if self.provider:
                    await self.provider.send_message(reply_msg)
                return reply_msg

        if kd and kd.pending_suggestions:
            result = kd.handle_suggestion_response(user_message.content)
            if result and result.get("action") in ("confirmed", "rejected"):
                from crewai.new_agent.models import Message as AgentMessage

                if result["action"] == "confirmed":
                    reply = f"Knowledge **{result['title']}** saved."
                else:
                    reply = "Knowledge suggestion dismissed."
                reply_msg = AgentMessage(
                    role="agent",
                    content=reply,
                    sender=self.agent.role,
                    conversation_id=conv_id,
                )
                self.conversation_history.append(user_message)
                self.conversation_history.append(reply_msg)
                if self.provider:
                    await self.provider.send_message(reply_msg)
                return reply_msg

        # Skill building: detect explicit instructions in user message.
        # Only trigger when the user clearly asks to remember/encode a procedure.
        # Use word-boundary-aware matching and skip if a suggestion is already pending.
        if (
            skill_builder
            and self.agent.settings.can_build_skills
            and not skill_builder.pending_suggestions
        ):
            lower_content = user_message.content.lower().strip()
            _has_skill_word = _match_skill_trigger(lower_content, "skill")
            _triggered = (
                _match_skill_trigger(lower_content, "remember how to")
                or _match_skill_trigger(lower_content, "from now on")
                or _match_skill_trigger(lower_content, "remember this procedure")
                or _match_skill_trigger(lower_content, "remember this process")
                or (
                    _has_skill_word
                    and any(
                        _match_skill_trigger(lower_content, verb)
                        for verb in (
                            "create",
                            "make",
                            "save",
                            "build",
                            "encode",
                            "turn",
                            "convert",
                        )
                    )
                )
            )
            if _triggered:
                try:
                    suggestion = skill_builder.suggest_from_instruction(
                        user_message.content
                    )
                    if suggestion and self.provider:
                        from crewai.new_agent.models import (
                            Message as AgentMessage,
                            MessageAction,
                        )

                        text, actions_data = skill_builder.build_suggestion_message(
                            suggestion
                        )
                        actions = [MessageAction(**a) for a in actions_data]
                        hint_msg = AgentMessage(
                            role="agent",
                            content=text,
                            actions=actions,
                            sender=self.agent.role,
                            conversation_id=conv_id,
                        )
                        self.conversation_history.append(user_message)
                        self.conversation_history.append(hint_msg)
                        if self.provider:
                            await self.provider.send_message(hint_msg)
                        return hint_msg
                except Exception:
                    pass

        # Check if dreaming is due (non-blocking background task)
        dreaming = getattr(self.agent, "_dreaming_engine", None)
        if dreaming and dreaming.should_dream():
            await self._emit_status("dreaming", "Consolidating memories…")
            asyncio.ensure_future(dreaming.dream())

        # Planning: assess complexity and create plan if warranted
        planning = getattr(self.agent, "_planning_engine", None)
        if planning is not None:
            await self._emit_status("planning", "Assessing task complexity…")
            # GAP-49: Track tokens before planning
            _plan_tokens_before_in = self._turn_input_tokens
            _plan_tokens_before_out = self._turn_output_tokens
            plan = await planning.maybe_plan(user_message.content)
            if plan:
                plan_text = "Follow this execution plan:\n" + "\n".join(
                    f"{i + 1}. {step}" for i, step in enumerate(plan)
                )
                self.prompt_stack.add("plan", plan_text, source="planning_engine")
            # GAP-49: Record sub-action tokens for planning
            self._track_tokens_from_llm()
            plan_in = self._turn_input_tokens - _plan_tokens_before_in
            plan_out = self._turn_output_tokens - _plan_tokens_before_out
            if plan_in > 0 or plan_out > 0:
                _plan_model = getattr(self.agent._llm_instance, "model", "") or ""
                self._record_sub_action_token_usage(
                    "planning", _plan_model, plan_in, plan_out
                )

        llm_messages = self._build_llm_messages(user_message)

        callbacks: list[TokenCalcHandler] = [TokenCalcHandler()]
        self._proactive_summarize_messages(llm_messages, callbacks)

        all_tools = list(self.agent._resolved_tools or []) + list(
            self.agent._coworker_tools or []
        )

        # Add spawn tool if agent can spawn
        if (
            self.agent.settings.can_spawn_copies
            and self.agent.settings.max_spawn_depth >= 1
        ):
            from crewai.new_agent.spawn_tools import SpawnSubtaskTool

            spawn_tool = SpawnSubtaskTool(agent=self.agent)
            if not any(t.name == spawn_tool.name for t in all_tools):
                all_tools.append(spawn_tool)

        await self._emit_status("thinking", "Analyzing your request…")

        llm = self.agent._llm_instance
        if llm is None:
            raise ValueError("Agent has no LLM configured.")

        # Resolve function_calling_llm for tool-use iterations
        fc_llm = self._resolve_function_calling_llm()

        tool_llm = fc_llm or llm
        use_native_tools = (
            hasattr(tool_llm, "supports_function_calling")
            and callable(getattr(tool_llm, "supports_function_calling", None))
            and tool_llm.supports_function_calling()
            and all_tools
        )

        openai_tools: list[dict[str, Any]] | None = None
        available_functions: dict[str, Callable[..., Any]] = {}
        if use_native_tools:
            openai_tools, available_functions, self._tool_name_mapping = (
                convert_tools_to_openai_schema(all_tools)
            )

        iterations = 0
        response_text = ""
        _thinking_text = ""  # GAP-53: thinking output from LLM
        llm_model = getattr(llm, "model", "") or ""

        # GAP-27: Enable reasoning/thinking on the LLM if supported (once per agent)
        if (
            self.agent.settings.reasoning_enabled
            and hasattr(llm, "thinking")
            and not llm.thinking
        ):
            try:
                from crewai.llms.providers.anthropic.completion import (
                    AnthropicCompletion,
                    AnthropicThinkingConfig,
                )

                if isinstance(llm, AnthropicCompletion):
                    llm.thinking = AnthropicThinkingConfig(type="adaptive")
                    try:
                        model_info = await asyncio.to_thread(
                            llm._get_sync_client().models.retrieve,
                            getattr(llm, "model", ""),
                        )
                        if model_info.max_tokens:
                            llm.max_tokens = model_info.max_tokens
                    except Exception:
                        pass
                else:
                    llm.thinking = True
            except ImportError:
                llm.thinking = True

        while True:
            if has_reached_max_iterations(iterations, self.max_iter):
                response_text = "I've reached the maximum number of iterations. Here's what I have so far based on my analysis."
                break

            # GAP-32: Check execution time deadline
            if deadline and time.monotonic() > deadline:
                response_text = "I've reached the maximum execution time. Here's what I have so far."
                break

            try:
                active_llm = tool_llm if (openai_tools and iterations > 0) else llm
                active_model = getattr(active_llm, "model", "") or llm_model

                # GAP-03: Emit LLM call started event
                try:
                    from crewai.new_agent.events import NewAgentLLMCallStartedEvent

                    self._emit_event(
                        NewAgentLLMCallStartedEvent(
                            new_agent_id=str(self.agent.id),
                            model=active_model,
                        )
                    )
                except Exception:
                    pass

                llm_call_start = time.monotonic()
                answer = await aget_llm_response(
                    llm=active_llm,
                    messages=llm_messages,
                    callbacks=callbacks,
                    printer=_NullPrinter(),
                    tools=openai_tools,
                    verbose=self.verbose,
                )
                self._track_tokens_from_llm()
                llm_call_elapsed = int((time.monotonic() - llm_call_start) * 1000)
                callbacks = [TokenCalcHandler()]

                # GAP-03: Emit LLM call completed event
                try:
                    from crewai.new_agent.events import NewAgentLLMCallCompletedEvent

                    self._emit_event(
                        NewAgentLLMCallCompletedEvent(
                            new_agent_id=str(self.agent.id),
                            model=active_model,
                            input_tokens=self._turn_input_tokens,
                            output_tokens=self._turn_output_tokens,
                            response_time_ms=llm_call_elapsed,
                        )
                    )
                except Exception:
                    pass

            except Exception as e:
                # GAP-03: Emit LLM call failed event
                try:
                    from crewai.new_agent.events import NewAgentLLMCallFailedEvent

                    self._emit_event(
                        NewAgentLLMCallFailedEvent(
                            new_agent_id=str(self.agent.id),
                            error=str(e),
                        )
                    )
                except Exception:
                    pass

                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.agent.settings.respect_context_window,
                        printer=_NullPrinter(),
                        messages=llm_messages,
                        llm=llm,
                        callbacks=callbacks,
                        verbose=self.verbose,
                    )
                    try:
                        from crewai.new_agent.events import (
                            NewAgentContextSummarizedEvent,
                        )

                        self._emit_event(
                            NewAgentContextSummarizedEvent(
                                new_agent_id=str(self.agent.id),
                            )
                        )
                    except Exception:
                        pass
                    iterations += 1
                    continue
                raise

            if isinstance(answer, list) and answer and self._is_tool_call_list(answer):
                tool_result = await self._handle_tool_calls(
                    answer, available_functions, llm_messages
                )
                if tool_result is not None:
                    response_text = tool_result
                    break

                self._proactive_summarize_messages(llm_messages, callbacks)

                # GAP-21: Call step_callback at each iteration boundary
                if self.agent.step_callback:
                    self.agent.step_callback(
                        iterations, self._tools_used_this_turn, response_text
                    )

                iterations += 1
                continue

            if isinstance(answer, BaseModel):
                response_text = answer.model_dump_json()
            elif isinstance(answer, str):
                response_text = answer
            else:
                response_text = str(answer)

            # GAP-53: Extract thinking output if available
            _thinking_text = self._extract_thinking_output(answer)

            # GAP-21: Call step_callback after LLM response
            if self.agent.step_callback:
                self.agent.step_callback(
                    iterations, self._tools_used_this_turn, response_text
                )

            break

        response_text = await self._run_guardrail(response_text)

        if self.agent.settings.narration_guard:
            response_text = await self._check_narration(response_text)

        # Structured output parsing
        structured_output = None
        if self.agent.response_model is not None:
            structured_output = await self._parse_structured_output(response_text)

        elapsed_ms = int((time.monotonic() - self._turn_start_time) * 1000)

        metadata: dict[str, Any] = {}
        if structured_output is not None:
            metadata["structured_output"] = structured_output.model_dump()

        # GAP-49: Include sub-action token data in metadata
        if self._sub_action_tokens:
            metadata["sub_action_tokens"] = [
                t.model_dump(mode="json") for t in self._sub_action_tokens
            ]

        # GAP-25: Estimate cost based on model and token usage
        estimated_cost = self._estimate_cost(
            llm_model, self._turn_input_tokens, self._turn_output_tokens
        )

        agent_message = Message(
            conversation_id=user_message.conversation_id,
            role="agent",
            content=response_text,
            sender=self.agent.role,
            model=llm_model,
            input_tokens=self._turn_input_tokens,
            output_tokens=self._turn_output_tokens,
            cost=estimated_cost,
            response_time_ms=elapsed_ms,
            tools_used=self._tools_used_this_turn or None,
            delegations=self._delegations_this_turn or None,
            # GAP-67: Attach artifacts detected from tool results
            artifacts=self._turn_artifacts if self._turn_artifacts else None,
            metadata=metadata if metadata else None,
        )

        self.conversation_history.append(user_message)
        self.conversation_history.append(agent_message)

        if self.agent.settings.provenance_enabled:
            # GAP-53: Use thinking output as free reasoning for standard/detailed provenance
            reasoning = _thinking_text if _thinking_text else ""

            # GAP-09: Generate explicit reasoning for 'detailed' provenance level
            if not reasoning:
                reasoning = await self._maybe_generate_reasoning(
                    "response",
                    {"user_message": user_message.content},
                    response_text[:500],
                )
            # GAP-49: Track sub-action tokens for the reasoning generation call
            if (
                self.agent.settings.provenance_detail == "detailed"
                and reasoning
                and not _thinking_text
            ):
                self._track_tokens_from_llm()
                # The reasoning LLM call tokens are already tracked in _maybe_generate_reasoning
                # via _track_tokens_from_llm, but record as sub-action for accounting
                self._record_sub_action_token_usage("reasoning", llm_model, 0, 0)

            prov_entry = ProvenanceEntry(
                conversation_id=user_message.conversation_id,
                action="response",
                reasoning=reasoning,
                inputs={"user_message": user_message.content},
                outcome=response_text[:500],
                # GAP-102: Populate sources from tools used this turn
                sources=self._tools_used_this_turn[:]
                if self._tools_used_this_turn
                else None,
                confidence=1.0,
            )
            self.provenance_log.append(prov_entry)
            # GAP-89: Persist provenance to memory backend
            self._persist_provenance_to_memory(prov_entry)

        self._record_token_usage("message", llm_model)
        self._save_to_memory(user_message, agent_message)
        self._emit_event_message_sent(agent_message)

        if self.provider:
            await self.provider.send_message(agent_message)

        # GAP-13: Save history to provider after each turn
        if self.provider and hasattr(self.provider, "save_history"):
            self.provider.save_history(self.conversation_history)

        # GAP-50: Save provenance to provider after each turn
        if self.provider and hasattr(self.provider, "save_provenance"):
            try:
                self.provider.save_provenance(self.provenance_log)
            except Exception:
                pass

        # GAP-46: Telemetry execution_completed span
        try:
            if hasattr(self.agent, "_telemetry") and self.agent._telemetry:
                self.agent._telemetry.execution_completed(
                    span=_telemetry_span,
                    input_tokens=self._turn_input_tokens,
                    output_tokens=self._turn_output_tokens,
                    response_time_ms=elapsed_ms,
                )
        except Exception:
            pass

        # GAP-33: Emit checkpoint data
        self._emit_checkpoint()

        return agent_message

    def _emit_event_message_received(self, msg: Message) -> None:
        try:
            from crewai.new_agent.events import NewAgentMessageReceivedEvent

            self._emit_event(
                NewAgentMessageReceivedEvent(
                    conversation_id=msg.conversation_id,
                    new_agent_id=str(self.agent.id),
                    message_length=len(msg.content),
                )
            )
        except Exception:
            pass

    def _emit_event_message_sent(self, msg: Message) -> None:
        try:
            from crewai.new_agent.events import NewAgentMessageSentEvent

            self._emit_event(
                NewAgentMessageSentEvent(
                    conversation_id=msg.conversation_id,
                    new_agent_id=str(self.agent.id),
                    new_agent_role=self.agent.role,
                    input_tokens=msg.input_tokens or 0,
                    output_tokens=msg.output_tokens or 0,
                    response_time_ms=msg.response_time_ms or 0,
                    model=msg.model or "",
                )
            )
        except Exception:
            pass

    def _is_tool_call_list(self, response: list[Any]) -> bool:
        if not response:
            return False
        first = response[0]
        if hasattr(first, "function") or (
            isinstance(first, dict) and "function" in first
        ):
            return True
        if hasattr(first, "type") and getattr(first, "type", None) == "tool_use":
            return True
        if hasattr(first, "name") and hasattr(first, "input"):
            return True
        if isinstance(first, dict) and "name" in first and "input" in first:
            return True
        return False

    async def _handle_tool_calls(
        self,
        tool_calls: list[Any],
        available_functions: dict[str, Callable[..., Any]],
        llm_messages: list[LLMMessage],
    ) -> str | None:
        """Execute tool calls and append results to messages. Returns final answer if tool has result_as_answer."""
        from crewai.utilities.agent_utils import parse_tool_call_args

        for tool_call in tool_calls:
            func_name, func_args, call_id = self._parse_tool_call(tool_call)
            if func_name is None:
                continue

            original_tool = self._tool_name_mapping.get(func_name)
            self._tools_used_this_turn.append(func_name)

            # GAP-117: Emit "delegating" status for coworker tools, "using_tool" for others
            if func_name.startswith("delegate_to_"):
                coworker_label = func_name.replace("delegate_to_", "").replace("_", " ")
                await self._emit_status(
                    "delegating", f"Asking @{coworker_label}…", coworker=coworker_label
                )
            else:
                await self._emit_status(
                    "using_tool", f"Using {func_name}…", tool_name=func_name
                )

            # GAP-04: Emit tool usage started event
            try:
                from crewai.new_agent.events import NewAgentToolUsageStartedEvent

                self._emit_event(
                    NewAgentToolUsageStartedEvent(
                        new_agent_id=str(self.agent.id),
                        tool_name=func_name,
                    )
                )
            except Exception:
                pass

            # GAP-26: Check tool result cache before execution
            cached = False
            result_str = ""
            if self.agent.settings.cache_tool_results:
                cache_key = (
                    f"{func_name}:{json.dumps(func_args, sort_keys=True, default=str)}"
                )
                if cache_key in self._tool_cache:
                    result_str = self._tool_cache[cache_key]
                    cached = True

            if not cached:
                try:
                    parsed_result = parse_tool_call_args(
                        func_args, func_name, call_id or func_name, original_tool
                    )
                    parsed_args, parse_error = parsed_result
                    if parse_error is not None:
                        result = parse_error.get(
                            "result", f"Error parsing args for {func_name}"
                        )
                    elif isinstance(parsed_args, dict):
                        result = (
                            original_tool._run(**parsed_args)
                            if original_tool
                            else str(parsed_args)
                        )
                    else:
                        result = (
                            original_tool._run(parsed_args)
                            if original_tool
                            else str(parsed_args)
                        )

                    result_str = str(result) if result is not None else ""

                    # GAP-04: Emit tool usage completed event
                    try:
                        from crewai.new_agent.events import (
                            NewAgentToolUsageCompletedEvent,
                        )

                        self._emit_event(
                            NewAgentToolUsageCompletedEvent(
                                new_agent_id=str(self.agent.id),
                                tool_name=func_name,
                            )
                        )
                    except Exception:
                        pass
                    await self._emit_status(
                        "thinking", f"Processing {func_name} result…"
                    )

                    # GAP-26: Store result in cache
                    if self.agent.settings.cache_tool_results:
                        cache_key = f"{func_name}:{json.dumps(func_args, sort_keys=True, default=str)}"
                        self._tool_cache[cache_key] = result_str

                except Exception as e:
                    result_str = f"Error executing {func_name}: {e}"

                    # GAP-04: Emit tool usage failed event
                    try:
                        from crewai.new_agent.events import NewAgentToolUsageFailedEvent

                        self._emit_event(
                            NewAgentToolUsageFailedEvent(
                                new_agent_id=str(self.agent.id),
                                tool_name=func_name,
                                error=str(e),
                            )
                        )
                    except Exception:
                        pass

            if self.agent.settings.provenance_enabled:
                # GAP-52: Generate reasoning for tool call provenance when detail is "detailed"
                tool_reasoning = ""
                if self.agent.settings.provenance_detail == "detailed":
                    try:
                        tool_reasoning = await self._maybe_generate_reasoning(
                            "tool_call",
                            {"tool": func_name, "args": str(func_args)[:200]},
                            result_str[:500],
                        )
                    except Exception:
                        pass
                tool_prov_entry = ProvenanceEntry(
                    conversation_id=self.conversation_history[0].conversation_id
                    if self.conversation_history
                    else "",
                    action="tool_call",
                    reasoning=tool_reasoning,
                    inputs={"tool": func_name, "args": str(func_args)[:200]},
                    outcome=result_str[:500],
                    # GAP-102: Populate sources and confidence for tool call provenance
                    sources=[func_name],
                    confidence=1.0 if not result_str.startswith("Error") else 0.5,
                )
                self.provenance_log.append(tool_prov_entry)
                # GAP-89: Persist tool call provenance to memory
                self._persist_provenance_to_memory(tool_prov_entry)

            # GAP-67: Detect artifacts from tool results
            try:
                detected_artifacts = self._detect_artifacts(func_name, result_str)
                if detected_artifacts:
                    self._turn_artifacts.extend(detected_artifacts)
            except Exception:
                pass

            args_str = (
                json.dumps(func_args) if isinstance(func_args, dict) else str(func_args)
            )
            llm_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id or func_name,
                            "type": "function",
                            "function": {"name": func_name, "arguments": args_str},
                        }
                    ],
                }
            )
            llm_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id or func_name,
                    "content": result_str,
                }
            )

            # Evaluate tool result for knowledge discovery
            kd = getattr(self.agent, "_knowledge_discovery", None)
            if kd and result_str:
                suggestion = kd.evaluate_for_knowledge(func_name, result_str)
                if suggestion and self.provider:
                    try:
                        from crewai.new_agent.models import (
                            Message as AgentMessage,
                            MessageAction,
                        )

                        text, actions_data = kd.build_suggestion_message(suggestion)
                        actions = [MessageAction(**a) for a in actions_data]
                        hint_msg = AgentMessage(
                            role="agent",
                            content=text,
                            actions=actions,
                            sender=self.agent.role,
                            conversation_id=self.conversation_history[0].conversation_id
                            if self.conversation_history
                            else "",
                        )
                        import asyncio

                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.ensure_future(self.provider.send_message(hint_msg))
                        else:
                            loop.run_until_complete(
                                self.provider.send_message(hint_msg)
                            )
                    except Exception:
                        pass

            if original_tool and getattr(original_tool, "result_as_answer", False):
                return result_str

        return None

    def _parse_tool_call(self, tool_call: Any) -> tuple[str | None, Any, str | None]:
        """Parse a tool call into (func_name, args, call_id)."""
        if hasattr(tool_call, "function"):
            fn = tool_call.function
            return (
                getattr(fn, "name", None),
                getattr(fn, "arguments", "{}"),
                getattr(tool_call, "id", None),
            )
        if isinstance(tool_call, dict):
            if "function" in tool_call:
                fn = tool_call["function"]
                return fn.get("name"), fn.get("arguments", "{}"), tool_call.get("id")
            if "name" in tool_call:
                return (
                    tool_call["name"],
                    tool_call.get("input", "{}"),
                    tool_call.get("id"),
                )
        if hasattr(tool_call, "name"):
            return (
                tool_call.name,
                getattr(tool_call, "input", "{}"),
                getattr(tool_call, "id", None),
            )
        return None, None, None

    async def _spawn_copies(self, sub_tasks: list[str]) -> list[str]:
        """Spawn copies of this agent for parallel sub-tasks.

        Creates N stripped-down copies (no backstory, history, or memory) and
        runs them concurrently.  Copies cannot spawn further copies (depth guard).
        """
        from crewai.new_agent.models import AgentSettings
        from crewai.new_agent.new_agent import NewAgent

        settings = self.agent.settings
        max_spawns = settings.max_concurrent_spawns
        timeout = settings.spawn_timeout

        # Cap the number of sub-tasks
        capped_tasks = sub_tasks[:max_spawns]

        # Build settings for the copies — depth-guarded, no memory
        spawn_settings = AgentSettings(
            can_spawn_copies=False,
            max_spawn_depth=0,
            memory_enabled=False,
            provenance_enabled=settings.provenance_enabled,
            respect_context_window=settings.respect_context_window,
            cache_tool_results=settings.cache_tool_results,
            narration_guard=settings.narration_guard,
            narration_max_retries=settings.narration_max_retries,
        )

        copies: list[NewAgent] = []
        for subtask in capped_tasks:
            copy = NewAgent(
                role=self.agent.role,
                goal=subtask,
                backstory="",
                llm=self.agent.llm,
                tools=list(self.agent.tools),
                memory=False,
                settings=spawn_settings,
                verbose=self.agent.verbose,
            )
            copies.append(copy)

        # Emit spawn started events
        spawn_ids: list[str] = []
        conv_id = (
            self.conversation_history[0].conversation_id
            if self.conversation_history
            else ""
        )
        for i, subtask in enumerate(capped_tasks):
            spawn_id = f"spawn-{i + 1}-{id(copies[i])}"
            spawn_ids.append(spawn_id)
            try:
                from crewai.new_agent.events import NewAgentSpawnStartedEvent

                self._emit_event(
                    NewAgentSpawnStartedEvent(
                        new_agent_id=str(self.agent.id),
                        spawn_id=spawn_id,
                        parent_id=str(self.agent.id),
                        spawn_depth=1,
                    )
                )
            except Exception:
                pass

        # Run all copies concurrently with timeout
        async_tasks = [
            asyncio.wait_for(copy.amessage(subtask), timeout=timeout)
            for copy, subtask in zip(copies, capped_tasks)
        ]
        raw_results = await asyncio.gather(*async_tasks, return_exceptions=True)

        results: list[str] = []

        for i, r in enumerate(raw_results):
            if isinstance(r, asyncio.TimeoutError):
                result_text = f"[Subtask {i + 1}] Timed out after {timeout}s"
                try:
                    from crewai.new_agent.events import NewAgentSpawnFailedEvent

                    self._emit_event(
                        NewAgentSpawnFailedEvent(
                            new_agent_id=str(self.agent.id),
                            spawn_id=spawn_ids[i],
                            error=f"Timed out after {timeout}s",
                        )
                    )
                except Exception:
                    pass
            elif isinstance(r, Exception):
                result_text = f"[Subtask {i + 1}] Error: {r}"
                try:
                    from crewai.new_agent.events import NewAgentSpawnFailedEvent

                    self._emit_event(
                        NewAgentSpawnFailedEvent(
                            new_agent_id=str(self.agent.id),
                            spawn_id=spawn_ids[i],
                            error=str(r),
                        )
                    )
                except Exception:
                    pass
            else:
                result_text = f"[Subtask {i + 1}] {r.content}"
                try:
                    from crewai.new_agent.events import NewAgentSpawnCompletedEvent

                    self._emit_event(
                        NewAgentSpawnCompletedEvent(
                            new_agent_id=str(self.agent.id),
                            spawn_id=spawn_ids[i],
                        )
                    )
                except Exception:
                    pass

            results.append(result_text)

            # Log provenance for each spawn
            if self.agent.settings.provenance_enabled:
                self.provenance_log.append(
                    ProvenanceEntry(
                        conversation_id=conv_id,
                        action="spawn",
                        reasoning=f"Spawned copy {i + 1}/{len(capped_tasks)} for parallel sub-task",
                        inputs={"subtask": capped_tasks[i]},
                        outcome=result_text[:500],
                    )
                )

        return results

    # ── Narration guard ────────────────────────────────────────

    _NARRATION_PATTERNS: list[re.Pattern[str]] = [
        re.compile(p, re.IGNORECASE)
        for p in (
            r"\bI've updated\b",
            r"\bI created\b",
            r"\bI sent\b",
            r"\bDone\s*[—–-]\s*[Tt]he\b",
            r"\bI've completed\b",
            r"\bI deleted\b",
            r"\bI modified\b",
        )
    ]

    async def _check_narration(self, response_text: str) -> str:
        """Check if the agent claimed actions it didn't perform.

        When narration_guard is enabled, this compares action-claiming language
        in the response against the tools actually used this turn.  If the agent
        narrates actions without corresponding tool calls it is asked to retry.
        """

        def _has_action_claims(text: str) -> bool:
            return any(p.search(text) for p in self._NARRATION_PATTERNS)

        max_retries = self.agent.settings.narration_max_retries

        for attempt in range(max_retries):
            if not _has_action_claims(response_text):
                return response_text

            if self._tools_used_this_turn:
                # Tools were actually used — claims are legitimate
                return response_text

            # Narration detected: agent claims actions but no tools were called
            logger.info(
                "Narration guard triggered (attempt %d/%d): agent claimed actions without tool calls",
                attempt + 1,
                max_retries,
            )
            try:
                from crewai.new_agent.events import NewAgentNarrationGuardTriggeredEvent

                self._emit_event(
                    NewAgentNarrationGuardTriggeredEvent(
                        new_agent_id=str(self.agent.id),
                        retries=attempt + 1,
                    )
                )
            except Exception:
                pass

            nudge = (
                "Your response claims you performed actions, but no tools were "
                "actually called. Either use the appropriate tools or correct "
                "your response."
            )
            response_text = await self._regenerate_with_feedback(response_text, nudge)

        # Final check after all retries
        if _has_action_claims(response_text) and not self._tools_used_this_turn:
            logger.warning(
                "Narration guard: unresolved after %d retries, flagging as bailout",
                max_retries,
            )
            if self.agent.settings.provenance_enabled:
                conv_id = (
                    self.conversation_history[0].conversation_id
                    if self.conversation_history
                    else ""
                )
                self.provenance_log.append(
                    ProvenanceEntry(
                        conversation_id=conv_id,
                        action="narration_bailout",
                        reasoning="Agent claimed actions without tool calls; unresolved after retries",
                        inputs={"response_excerpt": response_text[:300]},
                        outcome="narration_bailout",
                    )
                )

        return response_text

    async def astream(self, user_message: Message) -> AsyncGenerator[str, None]:
        """Stream a response token by token.

        Enables streaming on the LLM, runs ainvoke() as a background task,
        and yields text chunks via LLMStreamChunkEvent subscription.
        All turn logic (tools, provenance, memory, guardrails) is handled
        by ainvoke() — no duplicated code paths.
        """
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.llm_events import LLMStreamChunkEvent

        chunk_queue: asyncio.Queue[str] = asyncio.Queue()

        def _on_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            if event.chunk and not event.tool_call:
                chunk_queue.put_nowait(event.chunk)

        crewai_event_bus.on(LLMStreamChunkEvent)(_on_stream_chunk)

        llm = self.agent._llm_instance
        _prev_stream = getattr(llm, "stream", False) if llm else False
        if llm:
            llm.stream = True

        invoke_task = asyncio.create_task(self.ainvoke(user_message))
        _streamed_chars = 0
        _last_status_time = time.monotonic()

        try:
            while not invoke_task.done():
                try:
                    chunk = await asyncio.wait_for(chunk_queue.get(), timeout=0.05)
                    _streamed_chars += len(chunk)
                    yield chunk

                    now = time.monotonic()
                    if now - _last_status_time >= 0.5:
                        _last_status_time = now
                        await self._emit_status("streaming")
                except asyncio.TimeoutError:
                    continue

            while not chunk_queue.empty():
                chunk = chunk_queue.get_nowait()
                _streamed_chars += len(chunk)
                yield chunk

            result = invoke_task.result()
            self._last_stream_result = result
            if _streamed_chars == 0 and result.content:
                yield result.content

        finally:
            crewai_event_bus.off(LLMStreamChunkEvent, _on_stream_chunk)
            if llm:
                llm.stream = _prev_stream
            if not invoke_task.done():
                invoke_task.cancel()
                try:
                    await invoke_task
                except (asyncio.CancelledError, Exception):
                    pass


class _NullPrinter:
    """Minimal printer that swallows output."""

    def print(self, *args: Any, **kwargs: Any) -> None:
        pass
