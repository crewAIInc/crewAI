"""Build delegation tools from coworker agents.

GAP-49: Token tracking for delegation sub-actions.
GAP-55: Delegation provenance summary appended to results.
"""

from __future__ import annotations

import asyncio
from collections import Counter
import logging
import time
from typing import Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.string_utils import sanitize_tool_name


logger = logging.getLogger(__name__)


def _emit_delegation_event(event_cls: type, **kwargs: Any) -> None:
    try:
        from crewai.events.event_bus import crewai_event_bus

        crewai_event_bus.emit(None, event_cls(**kwargs))
    except Exception:
        pass


def _build_provenance_summary(
    coworker: Any, cw_role: str, elapsed_ms: int, in_tokens: int, out_tokens: int
) -> str:
    """GAP-55: Build a brief summary of what the coworker did during delegation."""
    try:
        executor = getattr(coworker, "_executor", None)
        if executor is None:
            return ""

        provenance = getattr(executor, "provenance_log", [])
        if not provenance:
            return ""

        # Count tool calls by name
        tool_counts: Counter[str] = Counter()
        step_count = 0
        for entry in provenance:
            step_count += 1
            if entry.action == "tool_call":
                tool_name = (entry.inputs or {}).get("tool", "unknown")
                tool_counts[tool_name] += 1

        if not tool_counts and step_count <= 1:
            return ""

        # Format tool usage summary
        tool_parts = []
        for tool_name, count in tool_counts.most_common():
            if count > 1:
                tool_parts.append(f"{tool_name} ({count}x)")
            else:
                tool_parts.append(tool_name)

        tools_str = ", ".join(tool_parts) if tool_parts else "none"
        in_k = f"{in_tokens:,}" if in_tokens else "0"
        out_k = f"{out_tokens:,}" if out_tokens else "0"

        return (
            f"\n\n---\n"
            f"[Coworker: {cw_role} | Tools: {tools_str} | "
            f"Steps: {step_count} | Tokens: ↑{in_k} ↓{out_k}]"
        )
    except Exception:
        return ""


class DelegateToCoworkerArgs(BaseModel):
    """Arguments for delegating work to a coworker."""

    message: str = Field(
        description="The message/instruction to send to the coworker. Be specific about what you need."
    )
    fire_and_forget: bool = Field(
        default=False,
        description="MUST be false (default) to get the coworker's response. Only set true for background tasks where you don't need the result.",
    )


class DelegateToCoworkerTool(BaseTool):
    """Tool that delegates work to a specific coworker agent."""

    name: str = ""
    description: str = ""
    args_schema: type[BaseModel] = DelegateToCoworkerArgs
    coworker: Any = None
    coworker_source: str = "local"
    parent_agent: Any = None

    def __init__(
        self,
        coworker: Any,
        source: str = "local",
        parent_agent: Any = None,
        **kwargs: Any,
    ) -> None:
        cw_role = getattr(coworker, "role", "coworker")
        tool_name = sanitize_tool_name(f"delegate_to_{cw_role}")
        cw_goal = getattr(coworker, "goal", "")
        desc = (
            f"Delegate work to {cw_role}. "
            f"Their expertise: {cw_goal}. "
            f"Send them a clear message describing what you need."
        )
        super().__init__(
            name=tool_name,
            description=desc,
            coworker=coworker,
            coworker_source=source,
            parent_agent=parent_agent,
            **kwargs,
        )

    def _run(self, message: str, fire_and_forget: bool = False, **kwargs: Any) -> str:
        """Execute delegation to the coworker."""
        from crewai.new_agent.events import (
            NewAgentDelegationCompletedEvent,
            NewAgentDelegationFailedEvent,
            NewAgentDelegationStartedEvent,
            NewAgentFireAndForgetCompletedEvent,
            NewAgentFireAndForgetDispatchedEvent,
        )
        from crewai.new_agent.new_agent import NewAgent

        cw_role = getattr(self.coworker, "role", "unknown")
        parent_id = getattr(self.parent_agent, "id", "") if self.parent_agent else ""

        if self.parent_agent and getattr(self.parent_agent, "on_delegate", None):
            self.parent_agent.on_delegate(self.coworker, message)

        if not isinstance(self.coworker, NewAgent):
            return self._delegate_a2a(message)

        if fire_and_forget:
            _emit_delegation_event(
                NewAgentFireAndForgetDispatchedEvent,
                new_agent_id=parent_id,
                coworker_role=cw_role,
            )
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            def _bg_fire_and_forget() -> None:
                try:
                    self.coworker.message(message)
                finally:
                    _emit_delegation_event(
                        NewAgentFireAndForgetCompletedEvent,
                        new_agent_id=parent_id,
                        coworker_role=cw_role,
                    )

            if loop and loop.is_running():

                async def _async_ff() -> None:
                    try:
                        await self.coworker.amessage(message)
                    finally:
                        _emit_delegation_event(
                            NewAgentFireAndForgetCompletedEvent,
                            new_agent_id=parent_id,
                            coworker_role=cw_role,
                        )

                loop.create_task(_async_ff())
            else:
                import threading

                threading.Thread(target=_bg_fire_and_forget, daemon=True).start()
            return f"Work delegated to {cw_role}. They are working on it in the background."

        _emit_delegation_event(
            NewAgentDelegationStartedEvent,
            new_agent_id=parent_id,
            coworker_role=cw_role,
            delegation_mode="sync",
            coworker_source=self.coworker_source,
        )

        start = time.monotonic()
        try:
            response = self.coworker.message(message)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            in_tokens = getattr(response, "input_tokens", 0) or 0
            out_tokens = getattr(response, "output_tokens", 0) or 0
            tokens = in_tokens + out_tokens
            _emit_delegation_event(
                NewAgentDelegationCompletedEvent,
                new_agent_id=parent_id,
                coworker_role=cw_role,
                tokens_consumed=tokens,
                response_time_ms=elapsed_ms,
            )

            # GAP-49: Record token usage on the parent agent if available
            if self.parent_agent and tokens > 0:
                try:
                    from crewai.new_agent.models import TokenUsage

                    executor = getattr(self.parent_agent, "_executor", None)
                    if executor is not None:
                        executor._sub_action_tokens.append(
                            TokenUsage(
                                action="delegation",
                                agent_id=str(parent_id),
                                input_tokens=in_tokens,
                                output_tokens=out_tokens,
                                model=getattr(response, "model", "") or "",
                                delegation_target=cw_role,
                                coworker_source=self.coworker_source,
                            )
                        )
                except Exception:
                    pass

            # GAP-55: Build and append provenance summary
            result_content = response.content
            summary = _build_provenance_summary(
                self.coworker, cw_role, elapsed_ms, in_tokens, out_tokens
            )
            if summary:
                result_content += summary

            return result_content
        except Exception as e:
            _emit_delegation_event(
                NewAgentDelegationFailedEvent,
                new_agent_id=parent_id,
                coworker_role=cw_role,
                error=str(e),
            )
            raise

    async def _arun(
        self, message: str, fire_and_forget: bool = False, **kwargs: Any
    ) -> str:
        """Async delegation — avoids blocking the event loop."""
        from crewai.new_agent.events import (
            NewAgentDelegationCompletedEvent,
            NewAgentDelegationFailedEvent,
            NewAgentDelegationStartedEvent,
            NewAgentFireAndForgetCompletedEvent,
            NewAgentFireAndForgetDispatchedEvent,
        )
        from crewai.new_agent.new_agent import NewAgent

        cw_role = getattr(self.coworker, "role", "unknown")
        parent_id = getattr(self.parent_agent, "id", "") if self.parent_agent else ""

        if self.parent_agent and getattr(self.parent_agent, "on_delegate", None):
            self.parent_agent.on_delegate(self.coworker, message)

        if not isinstance(self.coworker, NewAgent):
            return self._delegate_a2a(message)

        if fire_and_forget:
            _emit_delegation_event(
                NewAgentFireAndForgetDispatchedEvent,
                new_agent_id=parent_id,
                coworker_role=cw_role,
            )

            async def _async_ff() -> None:
                try:
                    await self.coworker.amessage(message)
                finally:
                    _emit_delegation_event(
                        NewAgentFireAndForgetCompletedEvent,
                        new_agent_id=parent_id,
                        coworker_role=cw_role,
                    )

            asyncio.get_running_loop().create_task(_async_ff())
            return f"Work delegated to {cw_role}. They are working on it in the background."

        _emit_delegation_event(
            NewAgentDelegationStartedEvent,
            new_agent_id=parent_id,
            coworker_role=cw_role,
            delegation_mode="sync",
            coworker_source=self.coworker_source,
        )

        start = time.monotonic()
        try:
            response = await self.coworker.amessage(message)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            in_tokens = getattr(response, "input_tokens", 0) or 0
            out_tokens = getattr(response, "output_tokens", 0) or 0
            tokens = in_tokens + out_tokens
            _emit_delegation_event(
                NewAgentDelegationCompletedEvent,
                new_agent_id=parent_id,
                coworker_role=cw_role,
                tokens_consumed=tokens,
                response_time_ms=elapsed_ms,
            )

            if self.parent_agent and tokens > 0:
                try:
                    from crewai.new_agent.models import TokenUsage

                    executor = getattr(self.parent_agent, "_executor", None)
                    if executor is not None:
                        executor._sub_action_tokens.append(
                            TokenUsage(
                                action="delegation",
                                agent_id=str(parent_id),
                                input_tokens=in_tokens,
                                output_tokens=out_tokens,
                                model=getattr(response, "model", "") or "",
                                delegation_target=cw_role,
                                coworker_source=self.coworker_source,
                            )
                        )
                except Exception:
                    pass

            result_content = response.content
            summary = _build_provenance_summary(
                self.coworker, cw_role, elapsed_ms, in_tokens, out_tokens
            )
            if summary:
                result_content += summary

            return result_content
        except Exception as e:
            _emit_delegation_event(
                NewAgentDelegationFailedEvent,
                new_agent_id=parent_id,
                coworker_role=cw_role,
                error=str(e),
            )
            raise

    def _delegate_a2a(self, message: str) -> str:
        """Delegate to an A2A remote coworker."""
        try:
            from crewai.a2a.client import A2AClient  # type: ignore[import-not-found]

            url = getattr(self.coworker, "url", None) or str(self.coworker)
            client = A2AClient(url=url)
            result = client.send_message(message)
            return str(result)
        except Exception as e:
            return f"A2A delegation failed: {e}"


class MultiDelegateArgs(BaseModel):
    """Arguments for delegating to multiple coworkers in parallel."""

    delegations: list[dict[str, str]] = Field(
        description=(
            "List of delegations. Each item is a dict with 'coworker' (role name) "
            "and 'message' (instruction to send). All coworkers run in parallel "
            "and results are collected."
        ),
    )


class MultiDelegateTool(BaseTool):
    """Tool that delegates work to multiple coworkers in parallel (sync)."""

    name: str = "delegate_to_multiple_coworkers"
    description: str = (
        "Delegate work to multiple coworkers simultaneously. "
        "Each coworker runs in parallel and all results are collected. "
        "Use when you need input from several coworkers to synthesize a response."
    )
    args_schema: type[BaseModel] = MultiDelegateArgs
    coworker_map: dict[str, Any] = Field(default_factory=dict)

    def _run(self, delegations: list[dict[str, str]], **kwargs: Any) -> str:
        """Execute parallel delegations to multiple coworkers."""
        from crewai.new_agent.new_agent import NewAgent

        tasks_to_run = []
        for d in delegations:
            cw_name = d.get("coworker", "")
            message = d.get("message", "")
            coworker = self.coworker_map.get(cw_name)
            if coworker is None:
                # Try matching by partial role name
                for role, cw in self.coworker_map.items():
                    if cw_name.lower() in role.lower():
                        coworker = cw
                        break
            if coworker is None or not isinstance(coworker, NewAgent):
                tasks_to_run.append((cw_name, message, None))
            else:
                tasks_to_run.append((cw_name, message, coworker))

        results: list[str] = []

        async def _run_all() -> list[str]:
            coros = []
            for cw_name, message, coworker in tasks_to_run:
                if coworker is None:
                    coros.append(_error_result(cw_name))
                else:
                    coros.append(coworker.amessage(message))
            raw_results = await asyncio.gather(*coros, return_exceptions=True)
            return [r for r in raw_results if not isinstance(r, BaseException)]

        async def _error_result(name: str) -> str:
            return f"[Error] Coworker '{name}' not found."

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                raw = pool.submit(asyncio.run, _run_all()).result()
        else:
            raw = asyncio.run(_run_all())

        for i, (cw_name, message, coworker) in enumerate(tasks_to_run):
            r = raw[i]
            if isinstance(r, Exception):
                results.append(f"[{cw_name}] Error: {r}")
            elif isinstance(r, str):
                results.append(f"[{cw_name}] {r}")
            else:
                content = getattr(r, "content", str(r))
                role = cw_name or f"Coworker {i + 1}"
                # GAP-55: Append provenance summary for each coworker
                in_tokens = getattr(r, "input_tokens", 0) or 0
                out_tokens = getattr(r, "output_tokens", 0) or 0
                if coworker is not None:
                    summary = _build_provenance_summary(
                        coworker, role, 0, in_tokens, out_tokens
                    )
                    if summary:
                        content += summary
                results.append(f"[{role}] {content}")

        return "\n\n".join(results)

    async def _arun(self, delegations: list[dict[str, str]], **kwargs: Any) -> str:
        """Async parallel delegation — avoids blocking the event loop."""
        from crewai.new_agent.new_agent import NewAgent

        tasks_to_run = []
        for d in delegations:
            cw_name = d.get("coworker", "")
            message = d.get("message", "")
            coworker = self.coworker_map.get(cw_name)
            if coworker is None:
                for role, cw in self.coworker_map.items():
                    if cw_name.lower() in role.lower():
                        coworker = cw
                        break
            if coworker is None or not isinstance(coworker, NewAgent):
                tasks_to_run.append((cw_name, message, None))
            else:
                tasks_to_run.append((cw_name, message, coworker))

        async def _error_result(name: str) -> str:
            return f"[Error] Coworker '{name}' not found."

        coros = []
        for cw_name, message, coworker in tasks_to_run:
            if coworker is None:
                coros.append(_error_result(cw_name))
            else:
                coros.append(coworker.amessage(message))
        raw = await asyncio.gather(*coros, return_exceptions=True)

        results: list[str] = []
        for i, (cw_name, message, coworker) in enumerate(tasks_to_run):
            r = raw[i]
            if isinstance(r, Exception):
                results.append(f"[{cw_name}] Error: {r}")
            elif isinstance(r, str):
                results.append(f"[{cw_name}] {r}")
            else:
                content = getattr(r, "content", str(r))
                role = cw_name or f"Coworker {i + 1}"
                in_tokens = getattr(r, "input_tokens", 0) or 0
                out_tokens = getattr(r, "output_tokens", 0) or 0
                if coworker is not None:
                    summary = _build_provenance_summary(
                        coworker, role, 0, in_tokens, out_tokens
                    )
                    if summary:
                        content += summary
                results.append(f"[{role}] {content}")

        return "\n\n".join(results)


def build_coworker_tools(
    coworkers: list[Any],
    parent_role: str = "",
    parent_agent: Any = None,
) -> list[BaseTool]:
    """Build delegation tools for a list of resolved coworkers."""
    tools: list[BaseTool] = []
    coworker_map: dict[str, Any] = {}
    for cw in coworkers:
        from crewai.new_agent.new_agent import NewAgent

        cw_role = getattr(cw, "role", "")
        if parent_role and cw_role == parent_role:
            continue

        if isinstance(cw, NewAgent):
            source = "amp" if getattr(cw, "_amp_resolved", False) else "local"
            tools.append(
                DelegateToCoworkerTool(
                    coworker=cw,
                    source=source,
                    parent_agent=parent_agent,
                )
            )
            coworker_map[cw.role] = cw
        else:
            source = "a2a"
            cw_url = getattr(cw, "url", None)
            if cw_url:
                tool_name = sanitize_tool_name(
                    f"delegate_to_a2a_{cw_url.split('/')[-1]}"
                )
                tools.append(
                    DelegateToCoworkerTool(
                        coworker=cw,
                        source=source,
                        parent_agent=parent_agent,
                    )
                )

    if len(coworker_map) > 1:
        tools.append(MultiDelegateTool(coworker_map=coworker_map))

    return tools
