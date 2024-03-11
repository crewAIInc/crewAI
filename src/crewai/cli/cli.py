import click

from .create_crew import create_crew


@click.group()
def crewai():
    """Top-level command group for crewai."""


@crewai.command()
@click.argument("project_name")
def create(project_name):
    """Create a new crew."""
    create_crew(project_name)


if __name__ == "__main__":
    crewai()
