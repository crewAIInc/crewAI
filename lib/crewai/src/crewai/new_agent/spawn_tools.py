"""Spawn tool — lets an agent spawn parallel copies of itself for sub-tasks.

GAP-57: Emits spawn started/completed/failed events.
GAP-58: Injects relevant parent memory into spawned copies.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool


logger = logging.getLogger(__name__)


def _emit_spawn_event(event_cls: type, **kwargs: Any) -> None:
    """Emit a spawn event on the event bus, swallowing errors."""
    try:
        from crewai.events.event_bus import crewai_event_bus

        crewai_event_bus.emit(None, event_cls(**kwargs))
    except Exception:
        pass


def _query_parent_memory(agent: Any, subtask: str, limit: int = 10) -> str:
    """GAP-58: Query the parent agent's memory for context relevant to the subtask.

    Returns a formatted context string, or empty string if unavailable.
    """
    try:
        memory = getattr(agent, "_memory_instance", None)
        if memory is None:
            return ""

        results = memory.recall(subtask, limit=limit)
        if not results:
            return ""

        lines: list[str] = []
        for m in results:
            content = getattr(m, "content", "") or getattr(
                getattr(m, "record", None), "content", ""
            )
            if content:
                lines.append(f"- {content}")

        if not lines:
            return ""

        return "Parent agent's relevant memory:\n" + "\n".join(lines)
    except Exception:
        return ""


class SpawnSubtaskArgs(BaseModel):
    """Arguments for spawning parallel sub-tasks."""

    subtasks: list[str] = Field(
        description="List of sub-task instructions to execute in parallel"
    )
    fire_and_forget: bool = Field(
        default=False,
        description="If true, dispatches subtasks in background without waiting for results.",
    )


class SpawnSubtaskTool(BaseTool):
    """Tool that spawns parallel copies of the agent for sub-tasks.

    Each copy receives the same tools but operates on a single sub-task
    with no backstory, history, or memory — just the instruction and tools.
    """

    name: str = "spawn_parallel_subtasks"
    description: str = (
        "Spawn parallel copies of yourself to handle multiple sub-tasks "
        "simultaneously. Each copy gets the same tools but focuses on one "
        "sub-task. Returns the collected results from all copies."
    )
    args_schema: type[BaseModel] = SpawnSubtaskArgs
    agent: Any = Field(default=None, exclude=True)

    def _run(
        self, subtasks: list[str], fire_and_forget: bool = False, **kwargs: Any
    ) -> str:
        """Execute parallel spawns synchronously."""
        from crewai.new_agent.new_agent import NewAgent

        if not isinstance(self.agent, NewAgent):
            return "Error: spawn tool requires a NewAgent instance."

        if not self.agent.settings.can_spawn_copies:
            return "Error: this agent is not allowed to spawn copies (can_spawn_copies=False)."

        if self.agent.settings.max_spawn_depth < 1:
            return "Error: spawn depth exceeded — copies cannot spawn further copies."

        settings = self.agent.settings
        max_spawns = settings.max_concurrent_spawns
        timeout = settings.spawn_timeout
        parent_id = str(self.agent.id)

        # Cap the number of sub-tasks
        if len(subtasks) > max_spawns:
            subtasks = subtasks[:max_spawns]

        # GAP-57: Generate spawn IDs and emit started events
        spawn_ids: list[str] = []
        for i, subtask in enumerate(subtasks):
            spawn_id = f"spawn-{uuid4().hex[:8]}-{i + 1}"
            spawn_ids.append(spawn_id)
            try:
                from crewai.new_agent.events import NewAgentSpawnStartedEvent

                _emit_spawn_event(
                    NewAgentSpawnStartedEvent,
                    new_agent_id=parent_id,
                    spawn_id=spawn_id,
                    parent_id=parent_id,
                    spawn_depth=1,
                )
            except Exception:
                pass

        spawn_start = time.monotonic()

        # Build stripped-down copies
        from crewai.new_agent.models import AgentSettings

        spawn_settings = AgentSettings(
            can_spawn_copies=False,
            max_spawn_depth=0,
            memory_enabled=True,  # Enable so copies can persist insights
            provenance_enabled=settings.provenance_enabled,
            respect_context_window=settings.respect_context_window,
            cache_tool_results=settings.cache_tool_results,
            narration_guard=settings.narration_guard,
            narration_max_retries=settings.narration_max_retries,
        )

        # GAP-58: Query parent memory for each subtask and build enriched messages
        enriched_messages: list[str] = []
        for subtask in subtasks:
            context = _query_parent_memory(self.agent, subtask)
            if context:
                enriched_messages.append(f"{context}\n\nTask: {subtask}")
            else:
                enriched_messages.append(subtask)

        copies: list[NewAgent] = []
        for subtask in subtasks:
            copy = NewAgent(
                role=self.agent.role,
                goal=subtask,
                backstory="",
                llm=self.agent.llm,
                tools=list(self.agent.tools),
                memory=True,  # Enable memory
                memory_scope=f"spawn-{parent_id}",  # Isolated scope
                settings=spawn_settings,
                verbose=self.agent.verbose,
            )
            copies.append(copy)

        # Fire-and-forget mode: start tasks in background threads and return immediately
        if fire_and_forget:
            import threading

            def _bg_spawn(copy: NewAgent, msg: str, sid: str) -> None:
                try:
                    copy.message(msg)
                    try:
                        from crewai.new_agent.events import NewAgentSpawnCompletedEvent

                        _emit_spawn_event(
                            NewAgentSpawnCompletedEvent,
                            new_agent_id=parent_id,
                            spawn_id=sid,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        from crewai.new_agent.events import NewAgentSpawnFailedEvent

                        _emit_spawn_event(
                            NewAgentSpawnFailedEvent,
                            new_agent_id=parent_id,
                            spawn_id=sid,
                            error=str(e),
                        )
                    except Exception:
                        pass

            for copy, msg, sid in zip(copies, enriched_messages, spawn_ids):
                threading.Thread(
                    target=_bg_spawn, args=(copy, msg, sid), daemon=True
                ).start()

            return f"Dispatched {len(copies)} subtask(s) in the background (fire-and-forget)."

        # Run in parallel
        async def _run_all() -> list[str]:
            tasks = [
                asyncio.wait_for(
                    copy.amessage(msg),
                    timeout=timeout,
                )
                for copy, msg in zip(copies, enriched_messages)
            ]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            output: list[str] = []
            for i, r in enumerate(raw_results):
                if isinstance(r, asyncio.TimeoutError):
                    output.append(f"[Subtask {i + 1}] Timed out after {timeout}s")
                    # GAP-57: Emit spawn failed event
                    try:
                        from crewai.new_agent.events import NewAgentSpawnFailedEvent

                        _emit_spawn_event(
                            NewAgentSpawnFailedEvent,
                            new_agent_id=parent_id,
                            spawn_id=spawn_ids[i],
                            error=f"Timed out after {timeout}s",
                        )
                    except Exception:
                        pass
                elif isinstance(r, Exception):
                    output.append(f"[Subtask {i + 1}] Error: {r}")
                    # GAP-57: Emit spawn failed event
                    try:
                        from crewai.new_agent.events import NewAgentSpawnFailedEvent

                        _emit_spawn_event(
                            NewAgentSpawnFailedEvent,
                            new_agent_id=parent_id,
                            spawn_id=spawn_ids[i],
                            error=str(r),
                        )
                    except Exception:
                        pass
                else:
                    output.append(f"[Subtask {i + 1}] {r.content}")
                    # GAP-57: Emit spawn completed event
                    try:
                        from crewai.new_agent.events import NewAgentSpawnCompletedEvent

                        _emit_spawn_event(
                            NewAgentSpawnCompletedEvent,
                            new_agent_id=parent_id,
                            spawn_id=spawn_ids[i],
                        )
                    except Exception:
                        pass
            return output

        # Handle event loop scenarios
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _run_all())
                results = future.result()
        else:
            results = asyncio.run(_run_all())

        self._log_spawn_provenance(subtasks, results, spawn_ids)
        return "\n\n".join(results)

    async def _arun(
        self, subtasks: list[str], fire_and_forget: bool = False, **kwargs: Any
    ) -> str:
        """Async spawn — avoids blocking the event loop."""
        from crewai.new_agent.new_agent import NewAgent

        if not isinstance(self.agent, NewAgent):
            return "Error: spawn tool requires a NewAgent instance."

        if not self.agent.settings.can_spawn_copies:
            return "Error: this agent is not allowed to spawn copies (can_spawn_copies=False)."

        if self.agent.settings.max_spawn_depth < 1:
            return "Error: spawn depth exceeded — copies cannot spawn further copies."

        settings = self.agent.settings
        max_spawns = settings.max_concurrent_spawns
        timeout = settings.spawn_timeout
        parent_id = str(self.agent.id)

        if len(subtasks) > max_spawns:
            subtasks = subtasks[:max_spawns]

        spawn_ids: list[str] = []
        for i, subtask in enumerate(subtasks):
            spawn_id = f"spawn-{uuid4().hex[:8]}-{i + 1}"
            spawn_ids.append(spawn_id)
            try:
                from crewai.new_agent.events import NewAgentSpawnStartedEvent

                _emit_spawn_event(
                    NewAgentSpawnStartedEvent,
                    new_agent_id=parent_id,
                    spawn_id=spawn_id,
                    parent_id=parent_id,
                    spawn_depth=1,
                )
            except Exception:
                pass

        from crewai.new_agent.models import AgentSettings as SpawnSettings

        spawn_settings = SpawnSettings(
            can_spawn_copies=False,
            max_spawn_depth=0,
            memory_enabled=True,
            provenance_enabled=settings.provenance_enabled,
            respect_context_window=settings.respect_context_window,
            cache_tool_results=settings.cache_tool_results,
            narration_guard=settings.narration_guard,
            narration_max_retries=settings.narration_max_retries,
        )

        enriched_messages: list[str] = []
        for subtask in subtasks:
            context = _query_parent_memory(self.agent, subtask)
            if context:
                enriched_messages.append(f"{context}\n\nTask: {subtask}")
            else:
                enriched_messages.append(subtask)

        copies: list[NewAgent] = []
        for subtask in subtasks:
            copy = NewAgent(
                role=self.agent.role,
                goal=subtask,
                backstory="",
                llm=self.agent.llm,
                tools=list(self.agent.tools),
                memory=True,
                memory_scope=f"spawn-{parent_id}",
                settings=spawn_settings,
                verbose=self.agent.verbose,
            )
            copies.append(copy)

        if fire_and_forget:
            for copy, msg, sid in zip(copies, enriched_messages, spawn_ids):

                async def _bg(c: NewAgent = copy, m: str = msg, s: str = sid) -> None:
                    try:
                        await c.amessage(m)
                        try:
                            from crewai.new_agent.events import (
                                NewAgentSpawnCompletedEvent,
                            )

                            _emit_spawn_event(
                                NewAgentSpawnCompletedEvent,
                                new_agent_id=parent_id,
                                spawn_id=s,
                            )
                        except Exception:
                            pass
                    except Exception as e:
                        try:
                            from crewai.new_agent.events import (
                                NewAgentSpawnFailedEvent,
                            )

                            _emit_spawn_event(
                                NewAgentSpawnFailedEvent,
                                new_agent_id=parent_id,
                                spawn_id=s,
                                error=str(e),
                            )
                        except Exception:
                            pass

                asyncio.get_running_loop().create_task(_bg())

            return f"Dispatched {len(copies)} subtask(s) in the background (fire-and-forget)."

        tasks = [
            asyncio.wait_for(copy.amessage(msg), timeout=timeout)
            for copy, msg in zip(copies, enriched_messages)
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[str] = []
        for i, r in enumerate(raw_results):
            if isinstance(r, asyncio.TimeoutError):
                results.append(f"[Subtask {i + 1}] Timed out after {timeout}s")
                try:
                    from crewai.new_agent.events import NewAgentSpawnFailedEvent

                    _emit_spawn_event(
                        NewAgentSpawnFailedEvent,
                        new_agent_id=parent_id,
                        spawn_id=spawn_ids[i],
                        error=f"Timed out after {timeout}s",
                    )
                except Exception:
                    pass
            elif isinstance(r, Exception):
                results.append(f"[Subtask {i + 1}] Error: {r}")
                try:
                    from crewai.new_agent.events import NewAgentSpawnFailedEvent

                    _emit_spawn_event(
                        NewAgentSpawnFailedEvent,
                        new_agent_id=parent_id,
                        spawn_id=spawn_ids[i],
                        error=str(r),
                    )
                except Exception:
                    pass
            else:
                results.append(f"[Subtask {i + 1}] {r.content}")
                try:
                    from crewai.new_agent.events import NewAgentSpawnCompletedEvent

                    _emit_spawn_event(
                        NewAgentSpawnCompletedEvent,
                        new_agent_id=parent_id,
                        spawn_id=spawn_ids[i],
                    )
                except Exception:
                    pass

        self._log_spawn_provenance(subtasks, results, spawn_ids)
        return "\n\n".join(results)

    def _log_spawn_provenance(
        self, subtasks: list[str], results: list[str], spawn_ids: list[str]
    ) -> None:
        if self.agent.settings.provenance_enabled and hasattr(self.agent, "_executor"):
            from crewai.new_agent.models import ProvenanceEntry

            executor = self.agent._executor
            conv_id = (
                executor.conversation_history[0].conversation_id
                if executor.conversation_history
                else ""
            )
            for i, (subtask, result) in enumerate(zip(subtasks, results)):
                executor.provenance_log.append(
                    ProvenanceEntry(
                        conversation_id=conv_id,
                        action="spawn",
                        reasoning=f"Spawned copy {i + 1}/{len(subtasks)} for parallel sub-task",
                        inputs={"subtask": subtask, "spawn_id": spawn_ids[i]},
                        outcome=result[:500],
                    )
                )
