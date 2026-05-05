"""``crewai skills`` subcommands for the self-improvement loop."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import click

from crewai.skills.self_improve import (
    ProposalStore,
    SkillReviewer,
    TraceStore,
    accept_proposal,
    reject_proposal,
)


_DEFAULT_REVIEW_MODEL = "anthropic/claude-haiku-4-5"


def _print_proposal_summary(p) -> None:
    kind = (
        f"PATCH→{p.target_skill}"
        if p.proposal_kind == "patch_existing"
        else "NEW"
    )
    click.echo(
        f"  {p.id}  {kind:<28} conf={p.confidence:.2f}  role={p.agent_role}"
    )
    click.echo(f"    {p.name}: {p.description}")


@click.group(name="skills")
def skills() -> None:
    """Manage agent skills."""


@skills.command(name="review")
@click.option(
    "--role",
    default=None,
    help="Limit review to one role (slug or full name). Default: all roles with traces.",
)
@click.option(
    "--model",
    default=_DEFAULT_REVIEW_MODEL,
    help=f"LiteLLM model id for the reviewer LLM. Default: {_DEFAULT_REVIEW_MODEL}.",
)
@click.option(
    "--min-traces",
    default=2,
    type=int,
    help="Minimum traces per role before review fires. Default: 2.",
)
@click.option(
    "--floor",
    default=0.6,
    type=float,
    help="Drop proposals below this confidence. Default: 0.6.",
)
def skills_review(role: str | None, model: str, min_traces: int, floor: float) -> None:
    """Mine accumulated traces for skill proposals.

    Reads role + goal from the trace metadata, calls the reviewer LLM, and
    persists each proposal that scores above ``--floor`` to the queue. Use
    ``crewai skills proposals list`` to see what came out.
    """
    from crewai import LLM

    trace_store = TraceStore()
    proposal_store = ProposalStore()

    if not trace_store.root.exists():
        click.echo(f"No traces yet at {trace_store.root}", err=True)
        raise SystemExit(1)

    # Group traces by role from disk; the role-slug dirs come from the store.
    by_role: dict[str, list] = defaultdict(list)
    role_dirs = (
        [trace_store.role_dir(role)] if role else sorted(trace_store.root.iterdir())
    )
    for d in role_dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            t = trace_store.load(path)
            by_role[t.agent_role].append(t)

    if not by_role:
        click.echo("No traces found." if role is None else f"No traces for role={role!r}.")
        return

    reviewer_llm = LLM(model=model)
    total_emitted = 0

    for agent_role, traces in by_role.items():
        if len(traces) < min_traces:
            click.echo(
                f"Skipping {agent_role!r}: {len(traces)} trace(s), "
                f"need {min_traces}."
            )
            continue

        agent_goal = next((t.agent_goal for t in traces if t.agent_goal), "")
        loaded_skills_seen = sorted({s for t in traces for s in t.loaded_skills})
        pending = [
            proposal_store.load(p) for p in proposal_store.list_for_role(agent_role)
        ]

        reviewer = SkillReviewer(
            agent_role=agent_role,
            agent_goal=agent_goal,
            llm=reviewer_llm,
            min_traces=min_traces,
            confidence_floor=floor,
        )
        click.echo(
            f"🧠 Reviewing {len(traces)} trace(s) for {agent_role!r} "
            f"(model={model}, pending={len(pending)})…"
        )
        proposals_out = reviewer.review(
            traces,
            loaded_skill_names=loaded_skills_seen,
            pending_proposals=pending,
        )

        for p in proposals_out:
            path = proposal_store.save(p)
            click.echo(f"  + {p.id}  conf={p.confidence:.2f}  {p.name}")
            click.echo(f"      → {path}")
        total_emitted += len(proposals_out)

    click.echo(
        f"\n✅ Done. {total_emitted} proposal(s) added to the queue. "
        f"Run `crewai skills proposals list` to view."
    )


@skills.group(name="proposals")
def proposals() -> None:
    """Manage skill proposals from the self-improvement reviewer."""


@proposals.command(name="list")
@click.option("--role", default=None, help="Filter by agent role (slug or full name).")
def proposals_list(role: str | None) -> None:
    """List pending proposals across all roles."""
    store = ProposalStore()
    items = store.list_for_role(role) if role else None

    if role:
        records = [store.load(p) for p in store.list_for_role(role)]
    else:
        records = store.list_all()

    if not records:
        click.echo("(no pending proposals)")
        return

    click.echo(f"{len(records)} proposal(s):")
    for p in records:
        _print_proposal_summary(p)


@proposals.command(name="show")
@click.argument("proposal_id")
def proposals_show(proposal_id: str) -> None:
    """Print the full body of a proposal."""
    store = ProposalStore()
    prop = store.find(proposal_id)
    if prop is None:
        click.echo(f"No proposal with id {proposal_id!r}", err=True)
        raise SystemExit(1)

    click.echo(f"id:           {prop.id}")
    click.echo(f"role:         {prop.agent_role}")
    click.echo(f"name:         {prop.name}")
    click.echo(f"description:  {prop.description}")
    click.echo(f"confidence:   {prop.confidence:.2f}")
    click.echo(f"kind:         {prop.proposal_kind}")
    if prop.target_skill:
        click.echo(f"target:       {prop.target_skill}")
    click.echo(f"derived from: {', '.join(prop.derived_from_runs)}")
    click.echo("\nrationale:")
    click.echo(prop.rationale)
    click.echo("\n--- SKILL.md body ---")
    click.echo(prop.body)


@proposals.command(name="accept")
@click.argument("proposal_id")
@click.option("--force", is_flag=True, help="Overwrite an existing skill of the same name.")
@click.option(
    "--skills-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    envvar="CREWAI_SELF_IMPROVE_SKILLS_DIR",
    help=(
        "Directory to write the SKILL.md to. Defaults to the env var "
        "CREWAI_SELF_IMPROVE_SKILLS_DIR, then to the platform data dir. "
        "Use a project-relative path (e.g. ./skills/learned) to keep "
        "accepted skills under version control — and pass the same path "
        "to Agent(self_improve=SelfImprovementConfig(skills_dir=...)) so "
        "the agent loads it on the next kickoff."
    ),
)
def proposals_accept(proposal_id: str, force: bool, skills_dir: Path | None) -> None:
    """Materialize a proposal as a live SKILL.md."""
    from crewai.skills.self_improve import SkillStore

    store = ProposalStore()
    prop = store.find(proposal_id)
    if prop is None:
        click.echo(f"No proposal with id {proposal_id!r}", err=True)
        raise SystemExit(1)
    skill_store = SkillStore(skills_root=skills_dir) if skills_dir else None
    try:
        path = accept_proposal(prop, force=force, skill_store=skill_store)
    except FileExistsError as e:
        click.echo(f"{e}", err=True)
        raise SystemExit(2) from None
    click.echo(f"✅ Accepted: {path}")
    click.echo(
        "    This skill will load on the next kickoff for any agent with "
        f"role={prop.agent_role!r} and self_improve enabled "
        "(make sure SelfImprovementConfig.skills_dir matches if you used --skills-dir)."
    )


@proposals.command(name="reject")
@click.argument("proposal_id")
def proposals_reject(proposal_id: str) -> None:
    """Drop a proposal from the queue without accepting."""
    store = ProposalStore()
    prop = store.find(proposal_id)
    if prop is None:
        click.echo(f"No proposal with id {proposal_id!r}", err=True)
        raise SystemExit(1)
    reject_proposal(prop)
    click.echo(f"🗑  Rejected: {prop.id}")


@proposals.command(name="tui")
def proposals_tui() -> None:
    """Open an interactive triage TUI for the proposals queue."""
    from crewai.cli.skill_proposals_tui import SkillProposalsTUI

    SkillProposalsTUI().run()
