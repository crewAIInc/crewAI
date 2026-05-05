"""LLM-driven skill reviewer.

Reads N ``RunTrace`` records, identifies recurring *approaches* (not
facts — those go to memory), and emits ``SkillProposal``s for human
review.

The reviewer never writes to the active skills directory itself; it only
populates the proposals queue. Acceptance is a separate, explicit step.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from crewai.llms.base_llm import BaseLLM
from crewai.skills.self_improve.models import RunTrace, SkillProposal


_TRACE_OUTPUT_TRUNCATE = 1500
_TRACE_TASK_TRUNCATE = 800


class _ReviewerOutput(BaseModel):
    """Envelope around the LLM's structured proposals.

    ``SkillProposal.agent_role`` and ``derived_from_runs`` are intentionally
    server-filled post-call; the LLM doesn't need to emit them.
    """

    model_config = ConfigDict(extra="ignore")

    proposals: list[SkillProposal] = Field(default_factory=list)


_SYSTEM_PROMPT = """\
You are reviewing execution traces from a CrewAI agent to decide what
*reusable approaches* are worth saving as agent skills.

The agent's role: {role}
The agent's goal: {goal}

You see {n} traces from past runs. Your job:

1. Identify recurring NON-DETERMINISTIC APPROACHES — patterns of how this
   agent thinks about a class of problems. Examples of approaches:
     - "When asked to scaffold a project, always start by listing required
       config files before writing code."
     - "If a tool returns empty results, rephrase the query before retrying."

2. DROP facts. Facts go to memory, not skills. Examples of facts:
     - "the user prefers dark mode"
     - "the API key for service X is in vault Y"
     - "this project uses pytest"

3. Drop one-off observations. A skill needs ≥2 traces showing the same
   approach. If only one trace shows it, omit.

4. If a proposal would patch an EXISTING loaded skill (listed below), set
   proposal_kind="patch_existing" and target_skill=<that skill name>.
   Otherwise leave proposal_kind="new".

5. Score confidence honestly. Below 0.6 will be auto-rejected, so don't
   pad. A pattern seen in 2/{n} traces with no contradictions ≈ 0.65; in
   all {n} traces with consistent outcome ≈ 0.85.

6. SKILL.md body should be markdown:
     - First line: short description in italics or as a sentence
     - "When to use this skill" section
     - "How to apply it" section with concrete steps
     - No invented facts or fabricated tool names

Loaded skills already available to this agent:
{loaded_skills}

Proposals already QUEUED for human review (do not re-propose these — a
prior reviewer round already surfaced them, they're awaiting accept/reject):
{pending_proposals}

If a recurring pattern matches a queued proposal semantically, simply
omit it from your output. Don't restate the same approach under a new
name. The human curator will resolve the queue.

Return a JSON object matching the schema. Empty proposals list is a valid
answer when no recurring approach is worth saving.
"""


_USER_PROMPT = """\
TRACES ({n}):

{traces}

Review the traces above and emit your proposals.
"""


def _format_trace(trace: RunTrace) -> str:
    """One block per trace, compact but enough signal."""
    task = (trace.task_description or "").strip()
    if len(task) > _TRACE_TASK_TRUNCATE:
        task = task[: _TRACE_TASK_TRUNCATE - 1] + "…"

    output = (trace.output_summary or "").strip()
    if len(output) > _TRACE_OUTPUT_TRUNCATE:
        output = output[: _TRACE_OUTPUT_TRUNCATE - 1] + "…"

    tool_lines: list[str] = []
    for t in trace.tool_calls:
        tag = "ok" if t.ok else "ERR"
        tool_lines.append(f"  [{tag}] {t.name}({t.args_summary})")
    tools_block = "\n".join(tool_lines) if tool_lines else "  (no tool calls)"

    return (
        f"--- {trace.id}  outcome={trace.outcome}  "
        f"max_iter_exhausted={trace.max_iter_exhausted}  "
        f"guardrail_passed={trace.guardrail_passed}\n"
        f"task:\n{task}\n\n"
        f"tool_calls ({len(trace.tool_calls)}):\n{tools_block}\n\n"
        f"output_summary:\n{output}\n"
    )


class SkillReviewer:
    """Synthesize ``SkillProposal``s from a batch of ``RunTrace``s.

    Stateless; the only state lives in the disk store. Pass an LLM (any
    ``BaseLLM`` instance — typically a small/cheap model like Haiku is
    enough since the reviewer just summarizes).
    """

    def __init__(
        self,
        *,
        agent_role: str,
        agent_goal: str,
        llm: BaseLLM,
        min_traces: int = 3,
        confidence_floor: float = 0.6,
    ) -> None:
        self.agent_role = agent_role
        self.agent_goal = agent_goal
        self.llm = llm
        self.min_traces = min_traces
        self.confidence_floor = confidence_floor

    def review(
        self,
        traces: list[RunTrace],
        *,
        loaded_skill_names: list[str] | None = None,
        pending_proposals: list[SkillProposal] | None = None,
    ) -> list[SkillProposal]:
        """Run the LLM review and return filtered proposals.

        Returns an empty list when ``len(traces) < self.min_traces`` so
        the reviewer is safe to call early — it just no-ops.

        Pass ``pending_proposals`` (typically the current contents of the
        ProposalStore for this role) so the reviewer doesn't re-emit
        semantic duplicates of items already awaiting human review.
        """
        if len(traces) < self.min_traces:
            return []

        loaded_skills_str = (
            "\n".join(f"  - {name}" for name in loaded_skill_names)
            if loaded_skill_names
            else "  (none)"
        )

        if pending_proposals:
            pending_str = "\n".join(
                f"  - {p.name}: {p.description}" for p in pending_proposals
            )
        else:
            pending_str = "  (none)"

        system = _SYSTEM_PROMPT.format(
            role=self.agent_role,
            goal=self.agent_goal,
            n=len(traces),
            loaded_skills=loaded_skills_str,
            pending_proposals=pending_str,
        )
        user = _USER_PROMPT.format(
            n=len(traces),
            traces="\n".join(_format_trace(t) for t in traces),
        )

        result = self.llm.call(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_model=_ReviewerOutput,
        )

        if not isinstance(result, _ReviewerOutput):
            return []

        run_ids = [t.id for t in traces]
        # Carry the agent's skills_dir from the latest trace forward so the
        # accept step writes to the same path the agent reads from. We pick
        # the *last* trace's value because configuration drift (a user
        # changing skills_dir between runs) should land at the most recent.
        skills_dir = next(
            (t.agent_skills_dir for t in reversed(traces) if t.agent_skills_dir),
            None,
        )
        return [
            p.model_copy(
                update={
                    "agent_role": self.agent_role,
                    "derived_from_runs": run_ids,
                    "skills_dir": skills_dir,
                }
            )
            for p in result.proposals
            if p.confidence >= self.confidence_floor
        ]
