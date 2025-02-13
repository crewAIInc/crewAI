"""CLI command for resetting memory storage."""
import logging
import subprocess
import sys
from typing import Optional

import click

from crewai.cli.utils import get_crew
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler

_logger = logging.getLogger(__name__)


def _log_error(message: str) -> None:
    """Log an error message."""
    _logger.error(message)
    click.echo(message, err=True)


def _reset_all_memories() -> None:
    """Reset all memory types."""
    ShortTermMemory().reset()
    EntityMemory().reset()
    LongTermMemory().reset()
    TaskOutputStorageHandler().reset()
    KnowledgeStorage().reset()


@click.command()
@click.option("-l", "--long", is_flag=True, help="Reset long-term memory")
@click.option("-s", "--short", is_flag=True, help="Reset short-term memory")
@click.option("-e", "--entity", is_flag=True, help="Reset entity memory")
@click.option("--knowledge", is_flag=True, help="Reset knowledge")
@click.option("-k", "--kickoff-outputs", is_flag=True, help="Reset kickoff outputs")
@click.option("-a", "--all", is_flag=True, help="Reset all memories")
def reset_memories_command(
    long: bool,
    short: bool,
    entity: bool,
    knowledge: bool,
    kickoff_outputs: bool,
    all: bool,
) -> int:
    """
    Reset the crew memories.

    Args:
        long: Reset long-term memory
        short: Reset short-term memory
        entity: Reset entity memory
        knowledge: Reset knowledge
        kickoff_outputs: Reset kickoff outputs
        all: Reset all memories
    """
    try:
        crew = get_crew()
        if all:
            if crew:
                crew.reset_memories(command_type="all")
            else:
                # When no crew exists, use default storage paths
                _reset_all_memories()
            click.echo("All memories have been reset.")
            return 0

        if not any([long, short, entity, kickoff_outputs, knowledge]):
            click.echo(
                "Please specify at least one memory type to reset using the appropriate flags."
            )
            return 0

        if not crew:
            click.echo("No crew found. Use --all to reset all memories.")
            return 0

        try:
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

            return 0
        except Exception as e:
            _log_error(f"An unexpected error occurred: {e}")
            raise click.exceptions.Exit(code=1)

    except subprocess.CalledProcessError as e:
        _log_error(f"An error occurred while resetting the memories: {e}")
        click.echo(e.output, err=True)
        raise click.exceptions.Exit(code=1)

    except Exception as e:
        _log_error(f"An unexpected error occurred: {e}")
        raise click.exceptions.Exit(code=1)
