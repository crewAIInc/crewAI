"""Knowledge Discovery — detect and suggest reusable knowledge for NewAgent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai.new_agent.new_agent import NewAgent

logger = logging.getLogger(__name__)


class KnowledgeDiscovery:
    """Identifies valuable information during conversations and suggests
    creating knowledge sources."""

    def __init__(self, agent: NewAgent):
        self.agent = agent
        self._pending_suggestions: list[dict[str, Any]] = []

    @property
    def pending_suggestions(self) -> list[dict[str, Any]]:
        return list(self._pending_suggestions)

    def evaluate_for_knowledge(
        self, tool_name: str, tool_result: str
    ) -> dict[str, Any] | None:
        """Evaluate a tool result for knowledge-worthiness.

        Returns a suggestion dict if the result is worth saving, None otherwise.
        """
        settings = getattr(self.agent.settings, "can_create_knowledge", True)
        if not settings:
            return None

        # Heuristic: results from search/scrape/read tools are often knowledge-worthy
        if len(tool_result) < 50:
            return None

        knowledge_tools = {
            "search_web",
            "scrape_url",
            "read_file",
            "search",
            "web_search",
            "read_website",
            "scrape",
            "fetch_url",
            "search_knowledge",
            "query_database",
            "read_document",
        }
        if tool_name.lower() not in knowledge_tools:
            return None

        # Extract a title from the first line or first sentence
        first_line = tool_result.split("\n", 1)[0].strip()
        if not first_line:
            first_line = tool_result[:100].strip()
        # Use first sentence if first line is very long
        if len(first_line) > 120:
            dot_pos = first_line.find(".")
            if dot_pos > 0:
                first_line = first_line[: dot_pos + 1]
            else:
                first_line = first_line[:100] + "..."
        title = f"{tool_name}: {first_line}" if first_line else tool_name

        suggestion = {
            "source_tool": tool_name,
            "content": tool_result[:2000],  # Truncate for suggestion
            "title": title,
            "status": "pending",
        }
        self._pending_suggestions.append(suggestion)

        self._emit_suggestion_event(suggestion)
        return suggestion

    def build_suggestion_message(
        self, suggestion: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return (conversational_text, actions) for a pending suggestion."""
        title = suggestion.get("title", "Untitled")
        content = suggestion.get("content", "")
        preview = content[:300] + ("..." if len(content) > 300 else "")

        text = (
            f"I found potentially useful information: **{title}**\n\n"
            f"```\n{preview}\n```\n\n"
            f"Would you like me to save this as a knowledge source? "
            f"You can say yes, no, or ask me to modify it first."
        )

        from crewai.new_agent.models import MessageAction

        actions = [
            MessageAction(
                action_id=f"knowledge-confirm-{title[:40]}",
                label="Approve",
                action_type="suggestion_confirm",
                payload={"type": "knowledge", "title": title},
            ),
            MessageAction(
                action_id=f"knowledge-reject-{title[:40]}",
                label="Dismiss",
                action_type="suggestion_reject",
                payload={"type": "knowledge", "title": title},
            ),
        ]
        return text, [a.model_dump() for a in actions]

    def handle_suggestion_response(self, user_text: str) -> dict[str, Any] | None:
        """Interpret a plain-text user response to a pending suggestion."""
        if not self._pending_suggestions:
            return None

        from crewai.new_agent.skill_builder import _detect_suggestion_intent

        intent = _detect_suggestion_intent(user_text)

        if intent == "confirm":
            suggestion = self._pending_suggestions[0]
            title = suggestion.get("title", "Untitled")
            if self.confirm_suggestion(0):
                self._pending_suggestions.pop(0)
                return {"action": "confirmed", "title": title}
            return {"action": "error", "title": title}

        if intent == "reject":
            suggestion = self._pending_suggestions[0]
            title = suggestion.get("title", "Untitled")
            self.reject_suggestion(0)
            self._pending_suggestions.pop(0)
            return {"action": "rejected", "title": title}

        return {"action": "ignored"}

    def confirm_suggestion(self, index: int) -> bool:
        """Confirm a knowledge suggestion and create the knowledge source."""
        if index < 0 or index >= len(self._pending_suggestions):
            return False

        suggestion = self._pending_suggestions[index]
        suggestion["status"] = "confirmed"

        try:
            from crewai.knowledge.source.string_knowledge_source import (
                StringKnowledgeSource,
            )

            source = StringKnowledgeSource(content=suggestion["content"])

            if self.agent.knowledge is not None:
                self.agent.knowledge.sources.append(source)
            else:
                self.agent.knowledge_sources.append(source)

            self._emit_confirmed_event()
            return True
        except Exception as e:
            logger.debug(f"Failed to create knowledge source: {e}")
            return False

    def reject_suggestion(self, index: int) -> None:
        """Reject a knowledge suggestion."""
        if 0 <= index < len(self._pending_suggestions):
            self._pending_suggestions[index]["status"] = "rejected"
            self._emit_rejected_event()

    def _emit_suggestion_event(self, suggestion: dict[str, Any]) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentKnowledgeSuggestedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentKnowledgeSuggestedEvent(
                    new_agent_id=str(self.agent.id),
                    source_type=suggestion.get("source_tool", ""),
                ),
            )
        except Exception:
            pass

    def _emit_confirmed_event(self) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentKnowledgeConfirmedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentKnowledgeConfirmedEvent(new_agent_id=str(self.agent.id)),
            )
        except Exception:
            pass

    def _emit_rejected_event(self) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentKnowledgeRejectedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentKnowledgeRejectedEvent(new_agent_id=str(self.agent.id)),
            )
        except Exception:
            pass
