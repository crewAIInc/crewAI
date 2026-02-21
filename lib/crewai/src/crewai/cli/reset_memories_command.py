import subprocess

import click

from crewai.cli.utils import get_crews, get_flows
from crewai.flow import Flow


def _reset_flow_memory(flow: Flow) -> None:
    """Reset memory for a single flow instance.

    Handles Memory, MemoryScope (both have .reset()), and MemorySlice
    (delegates to the underlying ._memory).  Silently succeeds when the
    storage directory does not exist yet (nothing to reset).

    Args:
        flow: The flow instance whose memory should be reset.
    """
    mem = flow.memory
    if mem is None:
        return
    try:
        if hasattr(mem, "reset"):
            mem.reset()
        elif hasattr(mem, "_memory") and hasattr(mem._memory, "reset"):
            mem._memory.reset()
    except (FileNotFoundError, OSError):
        pass


def reset_memories_command(
    memory: bool,
    knowledge: bool,
    agent_knowledge: bool,
    kickoff_outputs: bool,
    all: bool,
) -> None:
    """Reset the crew and flow memories.

    Args:
        memory: Whether to reset the unified memory.
        knowledge: Whether to reset the knowledge.
        agent_knowledge: Whether to reset the agents knowledge.
        kickoff_outputs: Whether to reset the latest kickoff task outputs.
        all: Whether to reset all memories.
    """
    try:
        if not any([memory, kickoff_outputs, knowledge, agent_knowledge, all]):
            click.echo(
                "No memory type specified. Please specify at least one type to reset."
            )
            return

        crews = get_crews()
        flows = get_flows()

        if not crews and not flows:
            raise ValueError("No crew or flow found.")

        for crew in crews:
            if all:
                crew.reset_memories(command_type="all")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Reset memories command has been completed."
                )
                continue
            if memory:
                crew.reset_memories(command_type="memory")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Memory has been reset."
                )
            if kickoff_outputs:
                crew.reset_memories(command_type="kickoff_outputs")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Latest Kickoff outputs stored has been reset."
                )
            if knowledge:
                crew.reset_memories(command_type="knowledge")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Knowledge has been reset."
                )
            if agent_knowledge:
                crew.reset_memories(command_type="agent_knowledge")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Agents knowledge has been reset."
                )

        for flow in flows:
            flow_name = flow.name or flow.__class__.__name__
            if all:
                _reset_flow_memory(flow)
                click.echo(
                    f"[Flow ({flow_name})] Reset memories command has been completed."
                )
                continue
            if memory:
                _reset_flow_memory(flow)
                click.echo(
                    f"[Flow ({flow_name})] Memory has been reset."
                )

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while resetting the memories: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
