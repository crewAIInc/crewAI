"""Wrapper for the reset-memories command.

Delegates to ``crewai.utilities.reset_memories`` when the full crewai
package is installed, otherwise prints a helpful error message.
"""

from __future__ import annotations

import click


def reset_memories_command(
    memory: bool,
    knowledge: bool,
    agent_knowledge: bool,
    kickoff_outputs: bool,
    all: bool,
) -> None:
    try:
        from crewai.utilities.reset_memories import (
            reset_memories_command as _reset,
        )
    except ImportError:
        click.secho(
            "The 'reset-memories' command requires the full crewai package.\n"
            "Install it with: pip install crewai",
            fg="red",
        )
        raise SystemExit(1) from None

    _reset(memory, knowledge, agent_knowledge, kickoff_outputs, all)
