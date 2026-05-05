"""Self-improving skills for crewAI agents.

When an Agent is configured with ``self_improve=True``, a TraceCollector
subscribes to the event bus during kickoff, captures tool calls + outcome
signals into a RunTrace, auto-grades the run, and persists the trace to
disk. Across many runs, a SkillReviewer mines those traces for recurring
approaches and emits SkillProposals for human review.
"""

from crewai.skills.self_improve.acceptance import accept_proposal, reject_proposal
from crewai.skills.self_improve.auto_grade import grade_trace
from crewai.skills.self_improve.collector import TraceCollector
from crewai.skills.self_improve.models import (
    Outcome,
    RunTrace,
    SelfImprovementConfig,
    SkillProposal,
    ToolCallRecord,
)
from crewai.skills.self_improve.reviewer import SkillReviewer
from crewai.skills.self_improve.storage import ProposalStore, SkillStore, TraceStore


__all__ = [
    "Outcome",
    "ProposalStore",
    "RunTrace",
    "SelfImprovementConfig",
    "SkillProposal",
    "SkillReviewer",
    "SkillStore",
    "ToolCallRecord",
    "TraceCollector",
    "TraceStore",
    "accept_proposal",
    "grade_trace",
    "reject_proposal",
]
