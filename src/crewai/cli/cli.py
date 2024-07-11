import click
import pkg_resources


from .create_crew import create_crew
from .train_crew import train_crew
from .replay_from_task import replay_task_command


@click.group()
def crewai():
    """Top-level command group for crewai."""


@crewai.command()
@click.argument("project_name")
def create(project_name):
    """Create a new crew."""
    create_crew(project_name)


@crewai.command()
@click.option(
    "--tools", is_flag=True, help="Show the installed version of crewai tools"
)
def version(tools):
    """Show the installed version of crewai."""
    crewai_version = pkg_resources.get_distribution("crewai").version
    click.echo(f"crewai version: {crewai_version}")

    if tools:
        try:
            tools_version = pkg_resources.get_distribution("crewai-tools").version
            click.echo(f"crewai tools version: {tools_version}")
        except pkg_resources.DistributionNotFound:
            click.echo("crewai tools not installed")


@crewai.command()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=5,
    help="Number of iterations to train the crew",
)
def train(n_iterations: int):
    """Train the crew."""
    click.echo(f"Training the crew for {n_iterations} iterations")
    train_crew(n_iterations)


@crewai.command()
@click.option(
    "-t",
    "--task_id",
    type=str,
    help="The task ID of the task to replay from. This will replay the task and all the tasks that were executed after it.",
)
def replay_from_task(task_id: str) -> None:
    """
    Replay the crew execution from a specific task.

    Args:
        task_id (str): The ID of the task to replay from.
    """
    try:
        click.echo(f"Replaying the crew from task {task_id}")
        replay_task_command(task_id)
    except Exception as e:
        click.echo(f"An error occurred while replaying: {e}", err=True)


if __name__ == "__main__":
    crewai()
