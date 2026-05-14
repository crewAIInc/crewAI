"""Dreaming — background memory consolidation for NewAgent.

GAP-48: Marks raw memories as processed so they are not re-processed.
GAP-49: Tracks token usage from the consolidation LLM call.
GAP-54: Scopes canonical memories (global / user / conversation) and only shares global ones.
GAP-62: Saves detected workflows as reusable JSON recipes.
GAP-80: Workflow user confirmation flow — pending list instead of auto-save.
GAP-81: Generate executable Python Flow code alongside JSON metadata.
GAP-82: match_workflow() to consult discovered flows during execution.
GAP-100: Persist scope classification with canonical memories.
GAP-101: Shared canonical memories tagged read-only.
GAP-112: Prune raw memories after dreaming consolidation.
GAP-113: Workflow detection threshold raised from 3 to 5.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai.new_agent.new_agent import NewAgent

logger = logging.getLogger(__name__)

# GAP-54: Scope constants for canonical memories
SCOPE_GLOBAL = "global"
SCOPE_USER = "user"
SCOPE_CONVERSATION = "conversation"

# GAP-54: Heuristic patterns for user-scoped memories
_USER_SCOPE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmy\s+(name|preference|email|account|setting)\b",
        r"\buser\s+prefer",
        r"\bpersonal\s+(preference|setting|detail)",
        r"\bI\s+(like|prefer|want|need|always|usually)\b",
        r"\b(his|her|their)\s+(name|preference|email|account)\b",
    )
]

# GAP-54: Patterns that indicate conversation-specific context
_CONVERSATION_SCOPE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bin this conversation\b",
        r"\bjust now\b",
        r"\bthis session\b",
        r"\bcurrent discussion\b",
    )
]


def _classify_scope(canonical_text: str) -> str:
    """Classify a canonical memory's scope using heuristics."""
    for pattern in _CONVERSATION_SCOPE_PATTERNS:
        if pattern.search(canonical_text):
            return SCOPE_CONVERSATION
    for pattern in _USER_SCOPE_PATTERNS:
        if pattern.search(canonical_text):
            return SCOPE_USER
    return SCOPE_GLOBAL


class DreamingEngine:
    """Consolidates raw memories into canonical insights."""

    def __init__(self, agent: NewAgent):
        self.agent = agent
        self._last_dreaming_time: datetime | None = None
        self._memories_since_last_dream: int = 0
        # GAP-48: Track processed memory IDs (persistent)
        self._processed_memory_ids: set[str] = set()
        self._cycle_count: int = 0
        self._load_processed_ids()
        # GAP-49: Token tracking for the last dream cycle
        self._last_cycle_tokens: Any = None
        # GAP-62: Discovered flow recipes from previous cycles
        self._discovered_flows: list[dict[str, Any]] = []
        self._load_discovered_flows()
        # GAP-80: Pending workflows awaiting user confirmation
        self._pending_workflows: list[dict[str, Any]] = []
        # GAP-122: Training feedback awaiting next consolidation cycle
        self._training_feedback: list[dict[str, Any]] = []

    # ── GAP-48: Persistent processed-memory tracking ──────────

    def _processed_ids_path(self) -> str:
        """Path to the JSON file persisting processed memory IDs."""
        agent_name = re.sub(r"[^a-zA-Z0-9_-]", "_", self.agent.role)[:64]
        base_dir = os.path.join(".crewai", "dreaming")
        return os.path.join(base_dir, f"{agent_name}_processed.json")

    def _load_processed_ids(self) -> None:
        """Load previously processed memory IDs from disk."""
        try:
            path = self._processed_ids_path()
            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                self._processed_memory_ids = set(data.get("ids", []))
                self._cycle_count = data.get("cycle_count", 0)
        except Exception:
            self._processed_memory_ids = set()

    def _save_processed_ids(self) -> None:
        """Persist processed memory IDs to disk."""
        try:
            path = self._processed_ids_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(
                    {
                        "ids": list(self._processed_memory_ids),
                        "cycle_count": self._cycle_count,
                    },
                    f,
                )
        except Exception as e:
            logger.debug(f"Failed to persist processed memory IDs: {e}")

    # ── GAP-62: Discovered flow persistence ───────────────────

    def _flows_manifest_path(self) -> str:
        return os.path.join(".crewai", "flows", "manifest.json")

    def _load_discovered_flows(self) -> None:
        """Load the flow manifest from disk."""
        try:
            path = self._flows_manifest_path()
            if os.path.exists(path):
                with open(path, "r") as f:
                    self._discovered_flows = json.load(f)
        except Exception:
            self._discovered_flows = []

    def _save_flow_recipe(self, workflow: dict[str, Any]) -> None:
        """GAP-62: Save a workflow as a reusable JSON recipe and register in manifest."""
        tools = workflow.get("tools", [])
        count = workflow.get("count", 0)
        if not tools:
            return

        try:
            flows_dir = os.path.join(".crewai", "flows")
            os.makedirs(flows_dir, exist_ok=True)

            # Generate a recipe name
            recipe_name = "_".join(tools[:5]).replace(" ", "_").lower()
            recipe_name = re.sub(r"[^a-zA-Z0-9_]", "", recipe_name)[:64]
            recipe_path = os.path.join(flows_dir, f"{recipe_name}.json")

            recipe = {
                "name": recipe_name,
                "tools": tools,
                "pattern_count": count,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "agent_role": self.agent.role,
                "description": f"Repeated pattern ({count}x): {' -> '.join(tools)}",
            }

            with open(recipe_path, "w") as f:
                json.dump(recipe, f, indent=2)

            # Update manifest
            manifest_path = self._flows_manifest_path()
            manifest: list[dict[str, Any]] = []
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, "r") as f:
                        manifest = json.load(f)
                except Exception:
                    manifest = []

            # Avoid duplicate entries
            if not any(entry.get("name") == recipe_name for entry in manifest):
                manifest.append(
                    {
                        "name": recipe_name,
                        "path": recipe_path,
                        "tools": tools,
                        "created_at": recipe["created_at"],
                    }
                )
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)

            self._discovered_flows = manifest
            logger.debug(f"Saved workflow recipe: {recipe_name}")
        except Exception as e:
            logger.debug(f"Failed to save workflow recipe: {e}")

    def _generate_flow_code(self, workflow: dict[str, Any]) -> str | None:
        """GAP-81: Generate executable Python Flow code for a workflow.

        Saves a ``.py`` file alongside the JSON metadata. The generated Flow
        is readable and editable by the user.

        Returns the file path on success, or None on failure.
        """
        tools = workflow.get("tools", [])
        if not tools:
            return None

        try:
            recipe_name = "_".join(tools[:5]).replace(" ", "_").lower()
            recipe_name = re.sub(r"[^a-zA-Z0-9_]", "", recipe_name)[:64]

            class_name = (
                "".join(word.capitalize() for word in recipe_name.split("_") if word)
                or "DetectedWorkflow"
            )

            # Build step methods
            steps: list[str] = []
            for i, tool_name in enumerate(tools):
                safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", tool_name)
                step_num = i + 1
                if i == 0:
                    decorator = "    @start()"
                else:
                    prev_safe = re.sub(r"[^a-zA-Z0-9_]", "_", tools[i - 1])
                    decorator = f'    @listen("step_{i}_{prev_safe}")'
                method = (
                    f"{decorator}\n"
                    f"    def step_{step_num}_{safe_name}(self):\n"
                    f'        """Calls {tool_name} tool."""\n'
                    f'        agent = self.state.get("agent")\n'
                    f'        if agent and "{tool_name}" in (agent.tools or {{}}):\n'
                    f'            result = agent.tools["{tool_name}"].run(\n'
                    f'                self.state.get("step_{step_num}_input", self.state.get("input", ""))\n'
                    f"            )\n"
                    f"        else:\n"
                    f"            result = None\n"
                    f'        self.state["step_{step_num}_result"] = result\n'
                    f"        return result"
                )
                steps.append(method)

            steps_code = "\n\n".join(steps)

            code = (
                f'"""Auto-generated Flow for workflow: {recipe_name}\n'
                f"\n"
                f"Tools: {' -> '.join(tools)}\n"
                f"Generated by CrewAI DreamingEngine.\n"
                f'"""\n'
                f"\n"
                f"from crewai.flow.flow import Flow, start, listen\n"
                f"\n"
                f"\n"
                f"class {class_name}(Flow):\n"
                f'    """Workflow: {" -> ".join(tools)}"""\n'
                f"\n"
                f"{steps_code}\n"
            )

            flows_dir = os.path.join(".crewai", "flows")
            os.makedirs(flows_dir, exist_ok=True)
            py_path = os.path.join(flows_dir, f"workflow_{recipe_name}.py")
            with open(py_path, "w") as f:
                f.write(code)

            logger.debug(f"Generated Flow code: {py_path}")
            return py_path
        except Exception as e:
            logger.debug(f"Failed to generate Flow code: {e}")
            return None

    # ── GAP-82: Match user messages against discovered workflows ──

    def match_workflow(self, user_message: str) -> dict[str, Any] | None:
        """Check if a user message matches a previously confirmed workflow.

        Uses keyword overlap between the message and workflow descriptions.
        Returns the matching workflow dict, or None if no match is found.
        """
        if not self._discovered_flows:
            return None
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "to",
            "and",
            "or",
            "of",
            "in",
            "for",
            "it",
            "on",
        }
        msg_lower = user_message.lower()
        msg_words = set(msg_lower.split()) - stop_words
        for flow in self._discovered_flows:
            desc = flow.get("description", "").lower()
            desc_words = set(desc.split()) - stop_words
            overlap = desc_words & msg_words
            if len(overlap) >= 3:
                return flow
        return None

    # ── GAP-112: Prune processed raw memories ────────────────────

    def _prune_processed_memories(self, processed_ids: set[str]) -> None:
        """Remove raw memories that have been consolidated into canonical insights.

        Keeps the most recent ``KEEP_RECENT`` memories as an audit trail.
        """
        memory = getattr(self.agent, "_memory_instance", None)
        if not memory:
            return
        try:
            KEEP_RECENT = 20
            prunable = sorted(processed_ids)
            if len(prunable) <= KEEP_RECENT:
                return  # Keep all if we haven't accumulated enough
            to_prune = prunable[:-KEEP_RECENT]  # Prune oldest, keep recent
            for mem_id in to_prune:
                try:
                    memory.delete(mem_id)
                except Exception:
                    pass
        except Exception:
            pass

    # ── GAP-122: Training feedback integration ─────────────────

    def add_training_feedback(self, feedback: str, task_context: str = "") -> None:
        """Receive training feedback for priority inclusion in the next dream cycle.

        Stored entries are injected into the consolidation prompt with higher
        weight so the agent learns from explicit user corrections faster.
        """
        self._training_feedback.append(
            {
                "feedback": feedback,
                "task_context": task_context,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        self.increment_memory_count()
        logger.debug("Training feedback received for agent '%s'", self.agent.role)

    # ── Core dreaming logic ───────────────────────────────────

    def should_dream(self) -> bool:
        """Check if dreaming should be triggered."""
        settings = self.agent.settings
        if not settings.self_improving:
            return False

        now = datetime.now(timezone.utc)

        # Time-based trigger
        if self._last_dreaming_time is not None:
            hours_since = (now - self._last_dreaming_time).total_seconds() / 3600
            if hours_since >= settings.dreaming_interval_hours:
                return True
        elif self._memories_since_last_dream >= settings.dreaming_trigger_threshold:
            # Threshold trigger on first run
            return True

        # Threshold trigger
        if self._memories_since_last_dream >= settings.dreaming_trigger_threshold:
            return True

        return False

    def increment_memory_count(self) -> None:
        self._memories_since_last_dream += 1

    async def dream(self) -> dict[str, Any]:
        """Run dreaming cycle. Returns summary of what was consolidated."""
        # Emit event
        self._emit_dreaming_started()
        self._cycle_count += 1

        result = {
            "memories_processed": 0,
            "canonical_created": 0,
            "workflows_detected": 0,
        }

        try:
            memory = getattr(self.agent, "_memory_instance", None)

            if memory is not None:
                # GAP-48: Filter out already-processed memories
                memories, memory_ids = self._get_recent_memories(memory)
                result["memories_processed"] = len(memories)

                if memories:
                    consolidated = await self._consolidate_memories(memories)
                    result["canonical_created"] = len(consolidated)

                    for canonical in consolidated:
                        # GAP-54 + GAP-100: Classify scope and persist with metadata
                        scope = _classify_scope(canonical)
                        try:
                            memory.remember(
                                canonical,
                                agent_role=self.agent.role,
                                importance=0.9,
                                metadata={
                                    "type": "canonical",
                                    "scope": scope,
                                    "dreaming_cycle": self._cycle_count,
                                },
                            )
                        except TypeError:
                            # Fallback if memory.remember() doesn't accept metadata
                            try:
                                memory.remember(
                                    canonical,
                                    agent_role=self.agent.role,
                                    importance=0.9,
                                )
                            except Exception as e:
                                logger.debug(f"Failed to save canonical memory: {e}")
                        except Exception as e:
                            logger.debug(f"Failed to save canonical memory: {e}")

                    # GAP-54: Only share global-scoped memories with coworkers
                    global_memories = [
                        c for c in consolidated if _classify_scope(c) == SCOPE_GLOBAL
                    ]
                    self._share_with_coworkers(global_memories)

                    # GAP-48: Mark these memories as processed
                    self._processed_memory_ids.update(memory_ids)
                    self._save_processed_ids()

                    # GAP-112: Prune raw memories that have been consolidated
                    self._prune_processed_memories(self._processed_memory_ids)

            # Detect workflow patterns from provenance (independent of memory)
            workflows = self._detect_workflows()
            result["workflows_detected"] = len(workflows)

            for wf in workflows:
                self._emit_workflow_detected(wf)
                # GAP-80: Propose only — no auto-save. User must confirm.
                self._propose_workflow(wf)

        except Exception as e:
            logger.warning(f"Dreaming cycle failed: {e}")

        # Always reset counters after a dreaming attempt
        self._last_dreaming_time = datetime.now(timezone.utc)
        self._memories_since_last_dream = 0

        self._emit_dreaming_completed(result)
        return result

    def _get_recent_memories(self, memory: Any) -> tuple[list[str], list[str]]:
        """Get memories accumulated since last dreaming cycle.

        GAP-48: Returns (memory_contents, memory_ids) filtering out already-processed IDs.
        """
        try:
            results = memory.recall("", limit=50)
            contents: list[str] = []
            ids: list[str] = []

            for m in results or []:
                # Try to extract a unique ID for this memory
                mem_id = getattr(m, "id", None) or getattr(
                    getattr(m, "record", None), "id", None
                )
                if mem_id is None:
                    # Use content hash as fallback ID
                    content = getattr(m, "content", "") or getattr(
                        getattr(m, "record", None), "content", ""
                    )
                    if content:
                        mem_id = str(hash(content))
                    else:
                        continue

                mem_id = str(mem_id)

                # GAP-48: Skip already-processed memories
                if mem_id in self._processed_memory_ids:
                    continue

                # GAP-101: Skip read-only shared memories during consolidation
                mem_metadata = (
                    getattr(m, "metadata", None)
                    or getattr(getattr(m, "record", None), "metadata", None)
                    or {}
                )
                if isinstance(mem_metadata, dict) and mem_metadata.get("read_only"):
                    continue

                content = getattr(m, "content", "") or getattr(
                    getattr(m, "record", None), "content", ""
                )
                # GAP-101: Also skip by tag prefix
                if content and content.startswith("[shared:read-only]"):
                    continue
                if content:
                    contents.append(content)
                    ids.append(mem_id)

            return contents, ids
        except Exception:
            return [], []

    def _get_dreaming_llm(self) -> Any:
        """Get the LLM to use for dreaming — dedicated or agent's default."""
        dreaming_llm_ref = self.agent.settings.dreaming_llm
        if dreaming_llm_ref is not None:
            from crewai.utilities.llm_utils import create_llm

            return create_llm(dreaming_llm_ref)
        return self.agent._llm_instance

    async def _consolidate_memories(self, memories: list[str]) -> list[str]:
        """Use LLM to consolidate raw memories into canonical insights."""
        llm = self._get_dreaming_llm()
        if llm is None:
            return []

        from crewai.utilities.agent_utils import (
            aget_llm_response,
            format_message_for_llm,
        )
        from crewai.utilities.types import LLMMessage

        memory_text = "\n".join(f"- {m}" for m in memories)

        # GAP-122: Include pending training feedback with higher priority
        training_section = ""
        if self._training_feedback:
            lines = []
            for entry in self._training_feedback:
                ctx = entry.get("task_context", "")
                fb = entry.get("feedback", "")
                if ctx:
                    lines.append(f"- [Context: {ctx}] {fb}")
                else:
                    lines.append(f"- {fb}")
            training_section = (
                "\n\nTraining feedback (HIGH PRIORITY — these are explicit user "
                "corrections and should be preserved as canonical insights):\n"
                + "\n".join(lines)
            )
            self._training_feedback.clear()

        prompt = (
            "You are analyzing a collection of raw memories from an AI agent's interactions. "
            "Your task is to consolidate these into canonical insights — key learnings, patterns, "
            "and important facts that should be retained long-term.\n\n"
            "Raw memories:\n"
            f"{memory_text}"
            f"{training_section}\n\n"
            "Instructions:\n"
            "1. Identify patterns, repeated themes, and key facts\n"
            "2. Consolidate redundant memories into single, clear statements\n"
            "3. Resolve any pronouns or vague references into specific, self-contained facts\n"
            "4. Drop any memories that are too vague or incomplete to be useful\n"
            "5. Output each canonical insight on its own line, prefixed with '- '\n"
            "6. Keep insights concise but self-contained\n"
            "7. Training feedback entries are high priority — always preserve them\n\n"
            "Canonical insights:"
        )

        messages: list[LLMMessage] = [format_message_for_llm(prompt, role="user")]

        try:
            from crewai.new_agent.executor import _NullPrinter

            response = await aget_llm_response(
                llm=llm,
                messages=messages,
                callbacks=[],
                printer=_NullPrinter(),
                verbose=False,
            )

            # GAP-49: Record token usage from the consolidation LLM call
            try:
                from crewai.new_agent.models import TokenUsage

                usage = getattr(llm, "_token_usage", None) or {}
                in_tokens = usage.get("prompt_tokens", 0)
                out_tokens = usage.get("completion_tokens", 0)
                model_name = getattr(llm, "model", "") or ""
                self._last_cycle_tokens = TokenUsage(
                    action="dreaming",
                    agent_id=str(self.agent.id),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    model=model_name,
                )
            except Exception:
                pass

            lines = str(response).strip().split("\n")
            canonical = []
            for line in lines:
                line = line.strip()
                if line.startswith("- "):
                    canonical.append(line[2:].strip())
                elif line:
                    canonical.append(line)
            return canonical
        except Exception as e:
            logger.debug(f"Memory consolidation LLM call failed: {e}")
            return []

    def _detect_workflows(self) -> list[dict[str, Any]]:
        """Detect repeated tool-call sequences in provenance logs."""
        executor = self.agent._executor
        if executor is None:
            return []

        provenance = executor.provenance_log
        tool_sequences: list[list[str]] = []
        current_sequence: list[str] = []

        for entry in provenance:
            if entry.action == "tool_call":
                tool_name = (entry.inputs or {}).get("tool", "")
                if tool_name:
                    current_sequence.append(tool_name)
            elif entry.action == "response":
                if len(current_sequence) >= 2:
                    tool_sequences.append(current_sequence)
                current_sequence = []

        if len(current_sequence) >= 2:
            tool_sequences.append(current_sequence)

        # Find repeated sequences (simplified — look for exact matches)
        from collections import Counter

        seq_counter = Counter(tuple(s) for s in tool_sequences)
        workflows = [
            {"tools": list(seq), "count": count}
            for seq, count in seq_counter.items()
            if count >= 5  # GAP-113: Must appear at least 5 times (plan threshold)
        ]

        return workflows

    def _share_with_coworkers(self, canonical_memories: list[str]) -> None:
        """Share general canonical memories with coworker agents as read-only.

        GAP-54: Only receives memories already filtered to global scope.
        GAP-101: Tags shared memories with read_only=True so they are protected.
        """
        coworkers = getattr(self.agent, "_resolved_coworkers", [])
        if not coworkers:
            return

        from crewai.new_agent.new_agent import NewAgent

        for cw in coworkers:
            if not isinstance(cw, NewAgent):
                continue
            cw_memory = getattr(cw, "_memory_instance", None)
            if cw_memory is None:
                continue
            for canonical in canonical_memories:
                try:
                    cw_memory.remember(
                        f"[shared:read-only][shared from {self.agent.role}] {canonical}",
                        agent_role=cw.role,
                        importance=0.7,
                        metadata={
                            "type": "canonical_shared",
                            "source_agent": self.agent.role,
                            "read_only": True,
                        },
                    )
                except TypeError:
                    # Fallback if remember() doesn't accept metadata kwarg
                    try:
                        cw_memory.remember(
                            f"[shared:read-only][shared from {self.agent.role}] {canonical}",
                            agent_role=cw.role,
                            importance=0.7,
                        )
                    except Exception:
                        pass
                except Exception:
                    pass

    def _propose_workflow(self, workflow: dict[str, Any]) -> None:
        """GAP-80: Add workflow to pending list and emit proposal event.

        Does NOT auto-save. The workflow stays pending until the user
        confirms via ``confirm_workflow()`` or rejects via ``reject_workflow()``.
        """
        tools = workflow.get("tools", [])
        count = workflow.get("count", 0)
        description = (
            f"Detected repeated pattern ({count}x): {' → '.join(tools)}. "
            f"This could be crystallized into an automated workflow."
        )
        workflow["description"] = description
        self._pending_workflows.append(workflow)
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentWorkflowProposedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentWorkflowProposedEvent(
                    new_agent_id=str(self.agent.id),
                    workflow_description=description,
                ),
            )
        except Exception:
            pass

    # ── GAP-80: User confirmation flow for workflows ─────────────

    def get_pending_workflows(self) -> list[dict[str, Any]]:
        """Return the list of workflows awaiting user confirmation."""
        return list(self._pending_workflows)

    def confirm_workflow(self, index: int) -> dict[str, Any] | None:
        """Confirm a pending workflow, saving it as a recipe and Flow code.

        Returns the confirmed workflow dict, or None if the index is invalid.
        """
        if index < 0 or index >= len(self._pending_workflows):
            return None
        workflow = self._pending_workflows.pop(index)
        self._save_flow_recipe(workflow)
        # GAP-81: Also generate executable Flow code
        self._generate_flow_code(workflow)
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentWorkflowConfirmedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentWorkflowConfirmedEvent(new_agent_id=str(self.agent.id)),
            )
        except Exception:
            pass
        return workflow

    def reject_workflow(self, index: int) -> dict[str, Any] | None:
        """Reject a pending workflow, removing it from the pending list.

        Returns the rejected workflow dict, or None if the index is invalid.
        """
        if index < 0 or index >= len(self._pending_workflows):
            return None
        return self._pending_workflows.pop(index)

    def _emit_dreaming_started(self) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentDreamingStartedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentDreamingStartedEvent(new_agent_id=str(self.agent.id)),
            )
        except Exception:
            pass

    def _emit_workflow_detected(self, workflow: dict[str, Any]) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentWorkflowDetectedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentWorkflowDetectedEvent(
                    new_agent_id=str(self.agent.id),
                    tools=workflow.get("tools", []),
                    count=workflow.get("count", 0),
                ),
            )
        except Exception:
            pass

    def _emit_dreaming_completed(self, result: dict[str, Any]) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentDreamingCompletedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentDreamingCompletedEvent(
                    new_agent_id=str(self.agent.id),
                    memories_processed=result.get("memories_processed", 0),
                    canonical_created=result.get("canonical_created", 0),
                    workflows_detected=result.get("workflows_detected", 0),
                ),
            )
        except Exception:
            pass
