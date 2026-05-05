"""Materialize an accepted ``SkillProposal`` as a live ``SKILL.md`` file.

Acceptance is the human checkpoint in the self-improvement loop: until a
proposal is accepted, it lives in the proposals queue and never affects
the agent. After acceptance, the SKILL.md lands at
``~/.crewai/skills/<role-slug>/<skill-name>/SKILL.md`` where the existing
skill loader discovers it on the next kickoff.
"""

from __future__ import annotations

import json
from pathlib import Path

from crewai.skills.self_improve.models import SkillProposal
from crewai.skills.self_improve.storage import ProposalStore, SkillStore


def _format_skill_md(proposal: SkillProposal) -> str:
    """Render the proposal as a SKILL.md document.

    The body is written verbatim — proposals already include their own
    markdown structure (title, sections). We only prepend the YAML
    frontmatter the loader requires. Both fields are JSON-quoted because
    JSON strings are valid YAML scalars and handle every special-char case
    safely.
    """
    body = proposal.body.lstrip()
    # If the LLM already emitted frontmatter (defensive), don't double it.
    if body.startswith("---"):
        return body if body.endswith("\n") else body + "\n"
    frontmatter = (
        f"---\n"
        f"name: {json.dumps(proposal.name)}\n"
        f"description: {json.dumps(proposal.description)}\n"
        f"---\n\n"
    )
    return frontmatter + body if body.endswith("\n") else frontmatter + body + "\n"


def accept_proposal(
    proposal: SkillProposal,
    *,
    force: bool = False,
    skill_store: SkillStore | None = None,
    proposal_store: ProposalStore | None = None,
) -> Path:
    """Write the proposal as a SKILL.md and remove it from the queue.

    Args:
        proposal: The proposal to materialize.
        force: When True, overwrite an existing SKILL.md at the target path.
        skill_store: Override for the live-skills store (test injection).
        proposal_store: Override for the proposals store (test injection).

    Returns:
        Path to the written ``SKILL.md``.

    Raises:
        FileExistsError: When a SKILL.md already exists at the target and
            ``force=False``.
    """
    skill_store = skill_store or SkillStore()
    proposal_store = proposal_store or ProposalStore()

    target_dir = skill_store.skill_dir(proposal.agent_role, proposal.name)
    skill_md = target_dir / "SKILL.md"

    if skill_md.exists() and not force:
        raise FileExistsError(
            f"{skill_md} already exists. Pass force=True to overwrite."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    skill_md.write_text(_format_skill_md(proposal), encoding="utf-8")

    # Once accepted, the proposal record is no longer the source of truth —
    # the SKILL.md is. Drop the queue entry so it doesn't show up in `list`.
    proposal_store.delete(proposal.id, proposal.agent_role)

    return skill_md


def reject_proposal(
    proposal: SkillProposal,
    *,
    proposal_store: ProposalStore | None = None,
) -> bool:
    """Delete a proposal from the queue. Returns True if it existed."""
    proposal_store = proposal_store or ProposalStore()
    return proposal_store.delete(proposal.id, proposal.agent_role)
