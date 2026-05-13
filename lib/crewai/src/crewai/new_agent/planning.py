"""Planning — execution plan creation for NewAgent.

GAP-49: Tracks token usage from plan creation and reasoning reconstruction LLM calls.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai.new_agent.new_agent import NewAgent

logger = logging.getLogger(__name__)


class PlanningEngine:
    """Creates execution plans for complex tasks."""

    def __init__(self, agent: NewAgent):
        self.agent = agent
        self._current_plan: list[str] | None = None
        # GAP-49: Token tracking for the last plan/reasoning call
        self._last_plan_tokens: Any = None

    @property
    def current_plan(self) -> list[str] | None:
        return self._current_plan

    async def maybe_plan(self, user_message: str) -> list[str] | None:
        """Decide if planning is needed and create a plan if so.

        Returns a list of plan steps, or None if no planning needed.
        """
        settings = self.agent.settings
        if not settings.planning_enabled:
            return None

        if settings.auto_plan:
            needs_plan = await self._assess_complexity(user_message)
            if not needs_plan:
                return None

        plan = await self._create_plan(user_message)
        self._current_plan = plan

        self._emit_planning_events(plan)
        return plan

    async def _assess_complexity(self, message: str) -> bool:
        """Use a heuristic to determine if a message needs planning."""
        # Simple heuristic: long messages, multiple questions, or explicit planning keywords
        complexity_indicators = [
            len(message) > 500,
            message.count("?") > 2,
            any(
                kw in message.lower()
                for kw in [
                    "step by step",
                    "plan",
                    "multiple",
                    "compare",
                    "analyze",
                    "research",
                    "comprehensive",
                    "detailed",
                    "all of",
                    "each of",
                    "every",
                ]
            ),
            message.count(",") > 4,
            message.count(" and ") > 3,
        ]
        return sum(complexity_indicators) >= 2

    async def _create_plan(self, message: str) -> list[str]:
        """Use LLM to create an execution plan."""
        llm = self.agent._llm_instance
        if llm is None:
            return []

        from crewai.utilities.agent_utils import (
            aget_llm_response,
            format_message_for_llm,
        )
        from crewai.utilities.types import LLMMessage

        tools_desc = ""
        if self.agent._resolved_tools:
            tools_desc = "Available tools: " + ", ".join(
                t.name for t in self.agent._resolved_tools
            )

        coworkers_desc = ""
        if self.agent._resolved_coworkers:
            coworkers_desc = "Available coworkers: " + ", ".join(
                getattr(cw, "role", str(cw)) for cw in self.agent._resolved_coworkers
            )

        prompt = (
            f"You are {self.agent.role}. Your goal: {self.agent.goal}\n\n"
            f"A user has asked: {message}\n\n"
            f"{tools_desc}\n{coworkers_desc}\n\n"
            "Create a concise execution plan. List each step on its own line, "
            "prefixed with a number and period (e.g., '1. Search for...'). "
            "Keep steps actionable and specific. Maximum 7 steps."
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

            # GAP-49: Record token usage from the planning LLM call
            try:
                from crewai.new_agent.models import TokenUsage

                usage = getattr(llm, "_token_usage", None) or {}
                in_tokens = usage.get("prompt_tokens", 0)
                out_tokens = usage.get("completion_tokens", 0)
                model_name = getattr(llm, "model", "") or ""
                self._last_plan_tokens = TokenUsage(
                    action="planning",
                    agent_id=str(self.agent.id),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    model=model_name,
                )
            except Exception:
                pass

            lines = str(response).strip().split("\n")
            steps = []
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Remove numbering prefix
                    clean = line.lstrip("0123456789.-) ").strip()
                    if clean:
                        steps.append(clean)
            return steps or [str(response).strip()]
        except Exception as e:
            logger.debug(f"Planning LLM call failed: {e}")
            return []

    async def reconstruct_reasoning(self, provenance_log: list[Any]) -> list[Any]:
        """Reconstruct reasoning for provenance entries with empty reasoning fields."""
        entries_without_reasoning = [e for e in provenance_log if not e.reasoning]
        if not entries_without_reasoning:
            return provenance_log

        llm = self.agent._llm_instance
        if llm is None:
            return provenance_log

        from crewai.utilities.agent_utils import (
            aget_llm_response,
            format_message_for_llm,
        )
        from crewai.utilities.types import LLMMessage

        log_text = "\n".join(
            f"- [{e.action}] inputs={e.inputs}, outcome={e.outcome}"
            for e in provenance_log
        )

        prompt = (
            f"You are analyzing the decision trace of an AI agent ({self.agent.role}).\n\n"
            f"Execution log:\n{log_text}\n\n"
            "For each action, explain WHY the agent took that action in 1-2 sentences. "
            "Output one reasoning per line in the same order as the log entries, prefixed with the action index (0-based):\n"
            "0: reason\n1: reason\n..."
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

            # GAP-49: Record token usage from the reasoning reconstruction call
            try:
                from crewai.new_agent.models import TokenUsage

                usage = getattr(llm, "_token_usage", None) or {}
                in_tokens = usage.get("prompt_tokens", 0)
                out_tokens = usage.get("completion_tokens", 0)
                model_name = getattr(llm, "model", "") or ""
                self._last_plan_tokens = TokenUsage(
                    action="planning",
                    agent_id=str(self.agent.id),
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    model=model_name,
                )
            except Exception:
                pass

            lines = str(response).strip().split("\n")
            for line in lines:
                line = line.strip()
                if ":" in line:
                    idx_str, reasoning = line.split(":", 1)
                    try:
                        idx = int(idx_str.strip())
                        if 0 <= idx < len(provenance_log):
                            provenance_log[idx].reasoning = reasoning.strip()
                    except (ValueError, IndexError):
                        continue
        except Exception:
            pass

        return provenance_log

    def _emit_planning_events(self, plan: list[str]) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import (
                NewAgentPlanningCompletedEvent,
                NewAgentPlanningStartedEvent,
            )

            crewai_event_bus.emit(
                self.agent,
                NewAgentPlanningStartedEvent(new_agent_id=str(self.agent.id)),
            )
            crewai_event_bus.emit(
                self.agent,
                NewAgentPlanningCompletedEvent(
                    new_agent_id=str(self.agent.id),
                    plan_steps_count=len(plan),
                ),
            )
        except Exception:
            pass
