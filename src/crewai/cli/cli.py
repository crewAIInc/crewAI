import os
from importlib.metadata import version as get_version
from typing import Optional, Tuple

import click

from crewai import (
    Crew,  # We'll assume a direct import of the Crew class or import from .somewhere
)
from crewai.cli.add_crew_to_flow import add_crew_to_flow
from crewai.cli.create_crew import create_crew
from crewai.cli.create_flow import create_flow
from crewai.cli.fetch_chat_llm import fetch_chat_llm
from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)

from .authentication.main import AuthenticationCommand
from .deploy.main import DeployCommand
from .evaluate_crew import evaluate_crew
from .fetch_crew_inputs import fetch_crew_inputs
from .install_crew import install_crew
from .kickoff_flow import kickoff_flow
from .plot_flow import plot_flow
from .replay_from_task import replay_task_command
from .reset_memories_command import reset_memories_command
from .run_crew import run_crew
from .tools.main import ToolCommand
from .train_crew import train_crew
from .update_crew import update_crew


@click.group()
@click.version_option(get_version("crewai"))
def crewai():
    """Top-level command group for crewai."""


@crewai.command()
@click.argument("type", type=click.Choice(["crew", "flow"]))
@click.argument("name")
@click.option("--provider", type=str, help="The provider to use for the crew")
@click.option("--skip_provider", is_flag=True, help="Skip provider validation")
def create(type, name, provider, skip_provider=False):
    """Create a new crew, or flow."""
    if type == "crew":
        create_crew(name, provider, skip_provider)
    elif type == "flow":
        create_flow(name)
    else:
        click.secho("Error: Invalid type. Must be 'crew' or 'flow'.", fg="red")


@crewai.command()
@click.option(
    "--tools", is_flag=True, help="Show the installed version of crewai tools"
)
def version(tools):
    """Show the installed version of crewai."""
    try:
        crewai_version = get_version("crewai")
    except Exception:
        crewai_version = "unknown version"
    click.echo(f"crewai version: {crewai_version}")

    if tools:
        try:
            tools_version = get_version("crewai")
            click.echo(f"crewai tools version: {tools_version}")
        except Exception:
            click.echo("crewai tools not installed")


@crewai.command()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=5,
    help="Number of iterations to train the crew",
)
@click.option(
    "-f",
    "--filename",
    type=str,
    default="trained_agents_data.pkl",
    help="Path to a custom file for training",
)
def train(n_iterations: int, filename: str):
    """Train the crew."""
    click.echo(f"Training the Crew for {n_iterations} iterations")
    train_crew(n_iterations, filename)


@crewai.command()
@click.option(
    "-t",
    "--task_id",
    type=str,
    help="Replay the crew from this task ID, including all subsequent tasks.",
)
def replay(task_id: str) -> None:
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


@crewai.command()
def log_tasks_outputs() -> None:
    """
    Retrieve your latest crew.kickoff() task outputs.
    """
    try:
        storage = KickoffTaskOutputsSQLiteStorage()
        tasks = storage.load()

        if not tasks:
            click.echo(
                "No task outputs found. Only crew kickoff task outputs are logged."
            )
            return

        for index, task in enumerate(tasks, 1):
            click.echo(f"Task {index}: {task['task_id']}")
            click.echo(f"Description: {task['expected_output']}")
            click.echo("------")

    except Exception as e:
        click.echo(f"An error occurred while logging task outputs: {e}", err=True)


@crewai.command()
@click.option("-l", "--long", is_flag=True, help="Reset LONG TERM memory")
@click.option("-s", "--short", is_flag=True, help="Reset SHORT TERM memory")
@click.option("-e", "--entities", is_flag=True, help="Reset ENTITIES memory")
@click.option("-kn", "--knowledge", is_flag=True, help="Reset KNOWLEDGE storage")
@click.option(
    "-k",
    "--kickoff-outputs",
    is_flag=True,
    help="Reset LATEST KICKOFF TASK OUTPUTS",
)
@click.option("-a", "--all", is_flag=True, help="Reset ALL memories")
def reset_memories(
    long: bool,
    short: bool,
    entities: bool,
    knowledge: bool,
    kickoff_outputs: bool,
    all: bool,
) -> None:
    """
    Reset the crew memories (long, short, entity, latest_crew_kickoff_ouputs). This will delete all the data saved.
    """
    try:
        if not all and not (long or short or entities or knowledge or kickoff_outputs):
            click.echo(
                "Please specify at least one memory type to reset using the appropriate flags."
            )
            return
        reset_memories_command(long, short, entities, knowledge, kickoff_outputs, all)
    except Exception as e:
        click.echo(f"An error occurred while resetting memories: {e}", err=True)


@crewai.command()
@click.option(
    "-n",
    "--n_iterations",
    type=int,
    default=3,
    help="Number of iterations to Test the crew",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="gpt-4o-mini",
    help="LLM Model to run the tests on the Crew. For now only accepting only OpenAI models.",
)
def test(n_iterations: int, model: str):
    """Test the crew and evaluate the results."""
    click.echo(f"Testing the crew for {n_iterations} iterations with model {model}")
    evaluate_crew(n_iterations, model)


@crewai.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.pass_context
def install(context):
    """Install the Crew."""
    install_crew(context.args)


@crewai.command()
def run():
    """Run the Crew."""
    click.echo("Running the Crew")
    run_crew()


@crewai.command()
def update():
    """Update the pyproject.toml of the Crew project to use uv."""
    update_crew()


@crewai.command()
def signup():
    """Sign Up/Login to CrewAI+."""
    AuthenticationCommand().signup()


@crewai.command()
def login():
    """Sign Up/Login to CrewAI+."""
    AuthenticationCommand().signup()


# DEPLOY CREWAI+ COMMANDS
@crewai.group()
def deploy():
    """Deploy the Crew CLI group."""
    pass


@crewai.group()
def tool():
    """Tool Repository related commands."""
    pass


@deploy.command(name="create")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
def deploy_create(yes: bool):
    """Create a Crew deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.create_crew(yes)


@deploy.command(name="list")
def deploy_list():
    """List all deployments."""
    deploy_cmd = DeployCommand()
    deploy_cmd.list_crews()


@deploy.command(name="push")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_push(uuid: Optional[str]):
    """Deploy the Crew."""
    deploy_cmd = DeployCommand()
    deploy_cmd.deploy(uuid=uuid)


@deploy.command(name="status")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deply_status(uuid: Optional[str]):
    """Get the status of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_status(uuid=uuid)


@deploy.command(name="logs")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_logs(uuid: Optional[str]):
    """Get the logs of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_logs(uuid=uuid)


@deploy.command(name="remove")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_remove(uuid: Optional[str]):
    """Remove a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.remove_crew(uuid=uuid)


@tool.command(name="create")
@click.argument("handle")
def tool_create(handle: str):
    tool_cmd = ToolCommand()
    tool_cmd.create(handle)


@tool.command(name="install")
@click.argument("handle")
def tool_install(handle: str):
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.install(handle)


@tool.command(name="publish")
@click.option(
    "--force",
    is_flag=True,
    show_default=True,
    default=False,
    help="Bypasses Git remote validations",
)
@click.option("--public", "is_public", flag_value=True, default=False)
@click.option("--private", "is_public", flag_value=False)
def tool_publish(is_public: bool, force: bool):
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.publish(is_public, force)


@crewai.group()
def flow():
    """Flow related commands."""
    pass


@flow.command(name="kickoff")
def flow_run():
    """Kickoff the Flow."""
    click.echo("Running the Flow")
    kickoff_flow()


@flow.command(name="plot")
def flow_plot():
    """Plot the Flow."""
    click.echo("Plotting the Flow")
    plot_flow()


@flow.command(name="add-crew")
@click.argument("crew_name")
def flow_add_crew(crew_name):
    """Add a crew to an existing flow."""
    click.echo(f"Adding crew {crew_name} to the flow")
    add_crew_to_flow(crew_name)


@crewai.command()
def chat():
    """
    Start a conversation with the Crew, collecting user-supplied inputs
    only if needed. This is a demo of a 'chat' flow.
    """
    click.secho("Welcome to CrewAI Chat!", fg="green")

    # --------------------------------------------------------------------------
    # 1) Attempt to fetch Crew inputs
    # --------------------------------------------------------------------------
    click.secho("Gathering crew inputs via `fetch_crew_inputs()`...", fg="cyan")
    try:
        crew_inputs = fetch_crew_inputs()
    except Exception as e:
        # If an error occurs, we print it and halt.
        click.secho(f"Error fetching crew inputs: {e}", fg="red")
        return

    # If crew_inputs is empty, that's fine. We'll proceed anyway.
    click.secho(
        f"Found placeholders (possibly empty): {sorted(list(crew_inputs))}", fg="yellow"
    )

    # --------------------------------------------------------------------------
    # 2) Retrieve the Chat LLM
    # --------------------------------------------------------------------------
    click.secho("Fetching the Chat LLM...", fg="cyan")
    try:
        chat_llm = fetch_chat_llm()
    except Exception as e:
        click.secho(f"Failed to retrieve Chat LLM: {e}", fg="red")
        return

    if not chat_llm:
        click.secho("No valid Chat LLM returned. Exiting.", fg="red")
        return

    # --------------------------------------------------------------------------
    # 3) Simple chat loop (demo)
    # --------------------------------------------------------------------------
    click.secho(
        "\nEntering interactive chat loop. Type 'exit' or Ctrl+C to quit.\n", fg="cyan"
    )

    while True:
        try:
            user_input = click.prompt("You", type=str)
            if user_input.strip().lower() in ["exit", "quit"]:
                click.echo("Exiting chat. Goodbye!")
                break

            # For demonstration, we'll call the LLM directly on the user input:
            response = chat_llm.generate(user_input)
            click.secho(f"\nAI: {response}\n", fg="green")

        except (KeyboardInterrupt, EOFError):
            click.echo("\nExiting chat. Goodbye!")
            break
        except Exception as e:
            click.secho(f"Error occurred while generating chat response: {e}", fg="red")
            break


def load_crew_and_find_inputs(file_path: str) -> Tuple[Optional[Crew], set]:
    """
    Attempt to load a Crew from the provided file path or default location.
    Then gather placeholders from tasks. Returns (crew, set_of_placeholders).
    """
    crew = None
    placeholders_found = set()

    # 1) If file_path is not provided, attempt to detect the default crew config.
    if not file_path:
        # This is naive detection logic.
        # A real implementation might search typical locations like ./
        # or src/<project_name>/config/ for a crew configuration.
        default_candidate = "crew.yaml"
        if os.path.exists(default_candidate):
            file_path = default_candidate

    # 2) Try to load the crew from file if file_path exists
    if file_path and os.path.isfile(file_path):
        # Pseudocode for loading a crew from file—may vary depending on how the user’s config is stored
        try:
            # For demonstration, we do something like:
            #   with open(file_path, "r") as f:
            #       content = f.read()
            #   crew_data = parse_yaml_crew(content)
            #   crew = Crew(**crew_data)
            # Placeholder logic below:
            crew = Crew(name="ExampleCrew")
        except Exception as e:
            click.secho(f"Error loading Crew from {file_path}: {e}", fg="red")
            raise e

    if crew:
        # 3) Inspect crew tasks for placeholders
        # For each Task, we gather placeholders used in description/expected_output
        for task in crew.tasks:
            placeholders_in_desc = extract_placeholders(task.description)
            placeholders_in_out = extract_placeholders(task.expected_output)
            placeholders_found.update(placeholders_in_desc)
            placeholders_found.update(placeholders_in_out)

    return crew, placeholders_found


def extract_placeholders(text: str) -> set:
    """
    Given a string, find all placeholders of the form {something} that might be used for input interpolation.
    This is a naive example—actual logic might do advanced parsing to avoid curly braces used in JSON.
    """
    import re

    if not text:
        return set()
    pattern = r"\{([a-zA-Z0-9_]+)\}"
    matches = re.findall(pattern, text)
    return set(matches)


if __name__ == "__main__":
    crewai()
