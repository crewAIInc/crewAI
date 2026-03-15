"""Backward-compatibility shim.

The CLI has moved to the ``crewai-cli`` package.  This module re-exports the
Click group so that ``python -m crewai.cli.cli`` or direct imports continue to
work when both packages are installed.
"""

from __future__ import annotations


try:
    from crewai_cli.cli import crewai
except ImportError:
    import click

    @click.group()
    def crewai() -> None:
        """Top-level command group for crewai."""

    @crewai.command()
    def _missing() -> None:
        click.secho(
            "The crewai CLI has moved to the crewai-cli package.\n"
            "Install it with: pip install crewai-cli   (or pip install crewai[cli])",
            fg="red",
        )
        raise SystemExit(1)


if __name__ == "__main__":
    crewai()
