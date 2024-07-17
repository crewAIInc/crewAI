import subprocess
import click

from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory


def reset_memories_command(long, short, entity, all) -> None:
    """
    Replay the crew execution from a specific task.

    Args:
      task_id (str): The ID of the task to replay from.
    """

    try:
        if all:
            ShortTermMemory().reset()
            EntityMemory().reset()
            LongTermMemory().reset()
            click.echo("All memories have been reset.")
        else:
            if long:
                LongTermMemory().reset()
                click.echo("Long term memory has been reset.")

            if short:
                ShortTermMemory().reset()
                click.echo("Short term memory has been reset.")
            if entity:
                EntityMemory().reset()
                click.echo("Short term memory has been reset.")

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while resetting the memories: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
