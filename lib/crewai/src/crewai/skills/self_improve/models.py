"""Data models for self-improving skills."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field


def _now() -> datetime:
    return datetime.now(UTC)


class SelfImprovementConfig(BaseModel):
    """Per-agent configuration for the self-improvement loop.

    All fields are optional with sensible defaults. Pass to ``Agent`` as
    ``self_improve=SelfImprovementConfig(skills_dir=Path("./skills/learned"))``
    to override; ``Agent(self_improve=True)`` uses defaults.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skills_dir: Path | None = Field(
        default=None,
        description=(
            "Where accepted SKILL.md files are written and auto-loaded from. "
            "When None, falls back to <db_storage_path>/self_improve/skills/. "
            "Set to a project-relative path (e.g. Path('./skills/learned')) "
            "to keep accepted skills under version control."
        ),
    )


Outcome = Literal["success", "failure", "unknown"]
ProposalKind = Literal["new", "patch_existing"]


def _new_id(prefix: str) -> str:
    """Generate a short id with a stable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ToolCallRecord(BaseModel):
    """One tool invocation within a run."""

    model_config = ConfigDict(frozen=True)

    name: str
    args_summary: str = Field(
        default="",
        description="Truncated string repr of args, suitable for clustering.",
    )
    ok: bool = True
    error: str | None = None
    duration_ms: int | None = None


class RunTrace(BaseModel):
    """One agent + task execution.

    Built incrementally by ``TraceCollector`` from event-bus events,
    finalized at agent completion, then persisted to disk.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: _new_id("run"))
    agent_id: str | None = None
    agent_role: str
    agent_goal: str = ""
    task_id: str | None = None
    task_description: str | None = None
    started_at: datetime = Field(default_factory=_now)
    ended_at: datetime | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    loaded_skills: list[str] = Field(default_factory=list)
    outcome: Outcome = "unknown"
    output_summary: str | None = None
    error: str | None = None
    max_iter_exhausted: bool = False
    guardrail_passed: bool | None = None

    @property
    def tool_error_count(self) -> int:
        return sum(1 for t in self.tool_calls if not t.ok)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)


class SkillProposal(BaseModel):
    """A proposed new or updated skill, awaiting human review.

    The reviewer LLM emits these directly via a thin envelope; the server
    stamps ``agent_role`` and ``derived_from_runs`` after the call, which is
    why those two have permissive defaults rather than being required.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: _new_id("prop"))
    agent_role: str = ""
    name: str
    description: str
    body: str
    rationale: str
    confidence: float = Field(ge=0.0, le=1.0)
    proposal_kind: ProposalKind = "new"
    target_skill: str | None = None
    derived_from_runs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
