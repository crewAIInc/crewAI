import subprocess

import click

from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler
from crewai.cli.utils import get_crew


def reset_memories_command(
    long,
    short,
    entity,
    knowledge,
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
    """

    try:
        crew = get_crew()
        if crew:
            if all:
                crew.reset_memories(command_type="all")
                click.echo("All memories have been reset.")
            else:
                if long:
                    crew.reset_memories(command_type="long")
                    click.echo("Long term memory has been reset.")

                if short:
                    crew.reset_memories(command_type="short")
                    click.echo("Short term memory has been reset.")
                if entity:
                    crew.reset_memories(command_type="entity")
                    click.echo("Entity memory has been reset.")
                if kickoff_outputs:
                    crew.reset_memories(command_type="kickoff_outputs")
                    click.echo("Latest Kickoff outputs stored has been reset.")
                if knowledge:
                    crew.reset_memories(command_type="knowledge")
                    click.echo("Knowledge has been reset.")

    except subprocess.CalledProcessError as e:
        click.echo(f"An error occurred while resetting the memories: {e}", err=True)
        click.echo(e.output, err=True)

    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)
