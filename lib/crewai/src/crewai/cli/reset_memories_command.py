import subprocess

import click

from crewai.cli.utils import get_crews


def reset_memories_command(
    long,
    short,
    entity,
    knowledge,
    agent_knowledge,
    kickoff_outputs,
    all,
) -> None:
    """
    Reset the crew memories.

    Args:
      long (bool): Whether to reset the long-term memory.
      short (bool): Whether to reset the short-term memory.
      entity (bool): Whether to reset the entity memory.
      kickoff_outputs (bool): Whether to reset the latest kickoff task outputs.
      all (bool): Whether to reset all memories.
      knowledge (bool): Whether to reset the knowledge.
      agent_knowledge (bool): Whether to reset the agents knowledge.
    """

    try:
        if not any(
            [long, short, entity, kickoff_outputs, knowledge, agent_knowledge, all]
        ):
            click.echo(
                "Geen geheugentype opgegeven. Geef ten minste één type op om te resetten."
            )
            return

        crews = get_crews()
        if not crews:
            raise ValueError("Geen crew gevonden.")
        for crew in crews:
            if all:
                crew.reset_memories(command_type="all")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Reset geheugen commando is voltooid."
                )
                continue
            if long:
                crew.reset_memories(command_type="long")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Lange-termijn geheugen is gereset."
                )
            if short:
                crew.reset_memories(command_type="short")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Korte-termijn geheugen is gereset."
                )
            if entity:
                crew.reset_memories(command_type="entity")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Entiteit geheugen is gereset."
                )
            if kickoff_outputs:
                crew.reset_memories(command_type="kickoff_outputs")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Laatste Kickoff outputs opgeslagen zijn gereset."
                )
            if knowledge:
                crew.reset_memories(command_type="knowledge")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Kennis is gereset."
                )
            if agent_knowledge:
                crew.reset_memories(command_type="agent_knowledge")
                click.echo(
                    f"[Crew ({crew.name if crew.name else crew.id})] Agenten kennis is gereset."
                )

    except subprocess.CalledProcessError as e:
        click.echo(f"Er is een fout opgetreden bij het resetten van de geheugens: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"Er is een onverwachte fout opgetreden: {e}", err=True)
