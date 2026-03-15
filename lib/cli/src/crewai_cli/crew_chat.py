"""Wrapper for the crew chat command.

Delegates to ``crewai.utilities.crew_chat.run_chat`` when the full crewai
package is installed, otherwise prints a helpful error message.
"""

from __future__ import annotations

import click


def run_chat() -> None:
    try:
        from crewai.utilities.crew_chat import run_chat as _run_chat
    except ImportError:
        click.secho(
            "The 'chat' command requires the full crewai package.\n"
            "Install it with: pip install crewai",
            fg="red",
        )
        raise SystemExit(1) from None

    _run_chat()
