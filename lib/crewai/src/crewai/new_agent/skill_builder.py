"""SkillBuilder — lets agents create and suggest SKILL.md files.

Mirrors KnowledgeDiscovery: detects patterns, builds pending suggestions,
emits events, and waits for user approval before writing to disk.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai.new_agent.new_agent import NewAgent
    from crewai.skills.models import Skill

logger = logging.getLogger(__name__)

_SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_SLUGIFY_RE = re.compile(r"[^a-z0-9]+")

_GENERATION_PROMPT = """\
You are generating a reusable skill definition for a CrewAI agent.
A skill is a set of instructions that tells the agent HOW to perform a procedure.

Source type: {source_type}
Input:
{source_text}

Generate a JSON object with exactly these fields:
- "name": a kebab-case identifier (lowercase letters, digits, hyphens only, max 64 chars)
- "description": a one-line description of what this skill does (max 200 chars)
- "instructions": markdown-formatted step-by-step instructions

Return ONLY the JSON object, no extra text.
"""


def _slugify(text: str, max_len: int = 64) -> str:
    slug = _SLUGIFY_RE.sub("-", text.lower().strip()).strip("-")
    return slug[:max_len]


_CONFIRM_WORDS = {
    "yes",
    "yep",
    "yeah",
    "sure",
    "approve",
    "confirmed",
    "accept",
    "lgtm",
}
_CONFIRM_PHRASES = {"go ahead", "save it", "sounds good", "looks good"}
_REJECT_WORDS = {"no", "nah", "nope", "reject", "decline"}
_REJECT_PHRASES = {"never mind", "no thanks", "don't save", "not now"}


def _detect_suggestion_intent(user_text: str) -> str:
    """Return 'confirm', 'reject', or 'ignore' for a user response.

    Only short responses (≤ 10 words) are treated as confirm/reject signals.
    Longer messages are always 'ignore' — they're conversational, not
    yes/no answers.  Single-word triggers must appear in the first two
    words; multi-word phrases can appear anywhere in the short text.
    """
    lower = user_text.lower().strip()
    words = lower.split()
    if not words:
        return "ignore"

    if len(words) > 10:
        return "ignore"

    leading = " ".join(words[:2])

    def _word_match(word: str, text: str) -> bool:
        return bool(re.search(rf"\b{re.escape(word)}\b(?!-)", text))

    for phrase in _CONFIRM_PHRASES:
        if phrase in lower:
            return "confirm"
    for word in _CONFIRM_WORDS:
        if _word_match(word, leading):
            return "confirm"

    for phrase in _REJECT_PHRASES:
        if phrase in lower:
            return "reject"
    for word in _REJECT_WORDS:
        if _word_match(word, leading):
            return "reject"

    return "ignore"


class SkillBuilder:
    """Builds, suggests, and manages auto-generated skills for a NewAgent."""

    def __init__(self, agent: NewAgent) -> None:
        self.agent = agent
        self._pending_suggestions: list[dict[str, Any]] = []
        self._active_skills: list[Skill] = []

        role_slug = _slugify(agent.role or str(agent.id))
        self._skills_dir = Path("agents") / role_slug / "skills"

        self._load_existing_skills()

    @property
    def pending_suggestions(self) -> list[dict[str, Any]]:
        return list(self._pending_suggestions)

    # ── Suggestion creation ──

    def suggest_skill(
        self,
        name: str,
        description: str,
        instructions: str,
        source: str,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a pending skill suggestion and emit an event."""
        if not self.agent.settings.can_build_skills:
            return {}

        name = _slugify(name)
        if not name:
            name = f"skill-{len(self._pending_suggestions) + 1}"

        if not _SKILL_NAME_RE.match(name):
            name = _slugify(name)

        for existing in self._active_skills:
            if existing.name == name:
                name = f"{name}-{len(self._pending_suggestions) + 1}"
                break

        suggestion: dict[str, Any] = {
            "name": name,
            "description": description[:200],
            "instructions": instructions,
            "source": source,
            "status": "pending",
            "metadata": metadata or {"auto-generated": "true"},
        }
        self._pending_suggestions.append(suggestion)
        self._emit_suggested_event(suggestion)
        return suggestion

    def build_suggestion_message(
        self, suggestion: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Return (conversational_text, actions) for a pending suggestion.

        Plain-text providers show just the text and let the user respond
        conversationally.  Rich providers (Slack, Teams) can render
        the actions as buttons or interactive cards.
        """
        name = suggestion.get("name", "skill")
        desc = suggestion.get("description", "")
        instructions = suggestion.get("instructions", "")
        preview = instructions[:300] + ("..." if len(instructions) > 300 else "")

        text = (
            f"I've identified a pattern that could be saved as a reusable skill:\n\n"
            f"**{name}** — {desc}\n\n"
            f"```\n{preview}\n```\n\n"
            f"Would you like me to save this skill? "
            f"You can say yes, no, or ask me to modify it first."
        )

        from crewai.new_agent.models import MessageAction

        actions = [
            MessageAction(
                action_id=f"skill-confirm-{name}",
                label="Approve",
                action_type="suggestion_confirm",
                payload={"type": "skill", "name": name},
            ),
            MessageAction(
                action_id=f"skill-reject-{name}",
                label="Dismiss",
                action_type="suggestion_reject",
                payload={"type": "skill", "name": name},
            ),
            MessageAction(
                action_id=f"skill-edit-{name}",
                label="Edit",
                action_type="suggestion_edit",
                payload={"type": "skill", "name": name},
            ),
        ]
        return text, [a.model_dump() for a in actions]

    def handle_suggestion_response(self, user_text: str) -> dict[str, Any] | None:
        """Interpret a plain-text user response to a pending suggestion.

        Returns a dict with ``{"action": "confirmed"|"rejected"|"ignored", ...}``
        or ``None`` if there are no pending suggestions.
        After 3 consecutive ignores the suggestion is auto-dismissed.
        """
        if not self._pending_suggestions:
            return None

        intent = _detect_suggestion_intent(user_text)

        if intent == "confirm":
            suggestion = self._pending_suggestions[0]
            if self.confirm_suggestion(0):
                return {"action": "confirmed", "name": suggestion["name"]}
            return {"action": "error", "name": suggestion["name"]}

        if intent == "reject":
            suggestion = self._pending_suggestions[0]
            name = suggestion["name"]
            self.reject_suggestion(0)
            return {"action": "rejected", "name": name}

        self._pending_suggestions[0]["_ignore_count"] = (
            self._pending_suggestions[0].get("_ignore_count", 0) + 1
        )
        if self._pending_suggestions[0]["_ignore_count"] >= 3:
            name = self._pending_suggestions[0]["name"]
            self.reject_suggestion(0)
            return {"action": "rejected", "name": name}

        return {"action": "ignored"}

    def suggest_from_instruction(self, user_text: str) -> dict[str, Any]:
        """Generate a skill suggestion from an explicit user instruction."""
        generated = self._generate_skill_content(user_text, "explicit-instruction")
        if not generated:
            return self.suggest_skill(
                name=_slugify(user_text[:60]),
                description=user_text[:200],
                instructions=user_text,
                source="explicit-instruction",
            )
        return self.suggest_skill(
            name=generated["name"],
            description=generated["description"],
            instructions=generated["instructions"],
            source="explicit-instruction",
        )

    def suggest_from_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Convert a DreamingEngine workflow into a skill suggestion."""
        tools = workflow.get("tools", [])
        count = workflow.get("count", 0)
        source_text = (
            f"Repeated tool sequence ({count}x): {' -> '.join(tools)}\n"
            + "\n".join(f"  Step {i + 1}: {t}" for i, t in enumerate(tools))
        )

        generated = self._generate_skill_content(source_text, "workflow-detection")
        if not generated:
            name = _slugify("-".join(tools[:4]))
            return self.suggest_skill(
                name=name or "workflow-skill",
                description=f"Automated workflow: {' -> '.join(tools)}",
                instructions=(
                    f"## Workflow (detected {count} times)\n\n"
                    + "\n".join(
                        f"{i + 1}. Use the **{t}** tool" for i, t in enumerate(tools)
                    )
                ),
                source="workflow-detection",
            )
        return self.suggest_skill(
            name=generated["name"],
            description=generated["description"],
            instructions=generated["instructions"],
            source="workflow-detection",
        )

    # ── Approval / rejection ──

    def confirm_suggestion(self, index: int) -> bool:
        """Approve a pending suggestion: write SKILL.md, load, and activate."""
        if index < 0 or index >= len(self._pending_suggestions):
            return False

        suggestion = self._pending_suggestions[index]
        if suggestion["status"] != "pending":
            return False

        name = suggestion["name"]
        description = suggestion["description"]
        instructions = suggestion["instructions"]
        metadata = suggestion.get("metadata", {})

        try:
            skill_path = self._write_skill_to_disk(
                name, description, instructions, metadata
            )
        except Exception as e:
            logger.warning(f"Failed to write skill '{name}': {e}")
            return False

        try:
            from crewai.skills.parser import (
                load_skill_instructions,
                load_skill_metadata,
            )

            skill = load_skill_metadata(skill_path)
            skill = load_skill_instructions(skill)
            self._active_skills.append(skill)
        except Exception as e:
            logger.warning(f"Failed to load skill '{name}' after writing: {e}")
            return False

        suggestion["status"] = "confirmed"
        self._pending_suggestions.pop(index)
        self._emit_confirmed_event(name)
        return True

    def reject_suggestion(self, index: int) -> None:
        if 0 <= index < len(self._pending_suggestions):
            self._pending_suggestions[index]["status"] = "rejected"
            name = self._pending_suggestions[index]["name"]
            self._pending_suggestions.pop(index)
            self._emit_rejected_event(name)

    def update_suggestion(self, index: int, instructions: str) -> bool:
        if 0 <= index < len(self._pending_suggestions):
            self._pending_suggestions[index]["instructions"] = instructions
            return True
        return False

    # ── Active skills ──

    def get_active_skills(self) -> list[Skill]:
        return list(self._active_skills)

    def format_skills_context(self) -> str:
        if not self._active_skills:
            return ""
        try:
            from crewai.skills.loader import format_skill_context

            sections = [format_skill_context(s) for s in self._active_skills]
            return "\n\n".join(sections)
        except Exception as e:
            logger.warning(f"Failed to format skills context: {e}")
            return ""

    # ── Disk I/O ──

    def _write_skill_to_disk(
        self,
        name: str,
        description: str,
        instructions: str,
        metadata: dict[str, str],
    ) -> Path:
        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        frontmatter_lines = [
            "---",
            f"name: {name}",
            f'description: "{description}"',
        ]
        if metadata:
            frontmatter_lines.append("metadata:")
            for k, v in metadata.items():
                frontmatter_lines.append(f'  {k}: "{v}"')
        frontmatter_lines.append("---")
        frontmatter_lines.append("")

        content = "\n".join(frontmatter_lines) + instructions
        (skill_dir / "SKILL.md").write_text(content)
        return skill_dir

    def _load_existing_skills(self) -> None:
        if not self._skills_dir.is_dir():
            return
        try:
            from crewai.skills.loader import activate_skill, discover_skills

            discovered = discover_skills(self._skills_dir)
            for skill in discovered:
                try:
                    activated = activate_skill(skill)
                    self._active_skills.append(activated)
                except Exception:
                    pass
        except Exception:
            pass

    # ── LLM skill generation ──

    def _generate_skill_content(
        self, source_text: str, source_type: str
    ) -> dict[str, Any] | None:
        llm = getattr(self.agent, "_llm_instance", None)
        if llm is None:
            return None

        prompt = _GENERATION_PROMPT.format(
            source_type=source_type,
            source_text=source_text,
        )

        try:
            from crewai.new_agent.executor import _NullPrinter
            from crewai.utilities.agent_utils import (
                format_message_for_llm,
                get_llm_response,
            )

            messages = [format_message_for_llm(prompt, role="user")]
            response = get_llm_response(
                llm=llm,
                messages=messages,
                callbacks=[],
                printer=_NullPrinter(),
                verbose=False,
            )

            text = str(response).strip()
            # Extract JSON from response (may be wrapped in ```json blocks)
            if "```" in text:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)

            data = json.loads(text)
            name = data.get("name", "")
            description = data.get("description", "")
            instructions = data.get("instructions", "")

            if not name or not instructions:
                return None

            return {
                "name": _slugify(name),
                "description": description[:200],
                "instructions": instructions,
            }
        except Exception as e:
            logger.debug(f"LLM skill generation failed: {e}")
            return None

    # ── Events ──

    def _emit_suggested_event(self, suggestion: dict[str, Any]) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentSkillSuggestedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentSkillSuggestedEvent(
                    new_agent_id=str(self.agent.id),
                    skill_name=suggestion.get("name", ""),
                    source_type=suggestion.get("source", ""),
                ),
            )
        except Exception:
            pass

    def _emit_confirmed_event(self, skill_name: str) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentSkillConfirmedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentSkillConfirmedEvent(
                    new_agent_id=str(self.agent.id),
                    skill_name=skill_name,
                ),
            )
        except Exception:
            pass

    def _emit_rejected_event(self, skill_name: str) -> None:
        try:
            from crewai.events.event_bus import crewai_event_bus
            from crewai.new_agent.events import NewAgentSkillRejectedEvent

            crewai_event_bus.emit(
                self.agent,
                event=NewAgentSkillRejectedEvent(
                    new_agent_id=str(self.agent.id),
                    skill_name=skill_name,
                ),
            )
        except Exception:
            pass
