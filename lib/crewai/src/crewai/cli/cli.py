from importlib.metadata import version as get_version
import os
import subprocess

import click

from crewai.cli.add_crew_to_flow import add_crew_to_flow
from crewai.cli.authentication.main import AuthenticationCommand
from crewai.cli.config import Settings
from crewai.cli.create_crew import create_crew
from crewai.cli.create_flow import create_flow
from crewai.cli.crew_chat import run_chat
from crewai.cli.deploy.main import DeployCommand
from crewai.cli.enterprise.main import EnterpriseConfigureCommand
from crewai.cli.evaluate_crew import evaluate_crew
from crewai.cli.install_crew import install_crew
from crewai.cli.kickoff_flow import kickoff_flow
from crewai.cli.organization.main import OrganizationCommand
from crewai.cli.plot_flow import plot_flow
from crewai.cli.replay_from_task import replay_task_command
from crewai.cli.reset_memories_command import reset_memories_command
from crewai.cli.run_crew import run_crew
from crewai.cli.settings.main import SettingsCommand
from crewai.cli.tools.main import ToolCommand
from crewai.cli.train_crew import train_crew
from crewai.cli.triggers.main import TriggersCommand
from crewai.cli.update_crew import update_crew
from crewai.cli.utils import build_env_with_tool_repository_credentials, read_toml
from crewai.memory.storage.kickoff_task_outputs_storage import (
    KickoffTaskOutputsSQLiteStorage,
)


@click.group()
@click.version_option(get_version("crewai"))
def crewai():
    """Top-level command group for crewai."""


@crewai.command(
    name="uv",
    context_settings=dict(
        ignore_unknown_options=True,
    ),
)
@click.argument("uv_args", nargs=-1, type=click.UNPROCESSED)
def uv(uv_args):
    """A wrapper around uv commands that adds custom tool authentication through env vars."""
    env = os.environ.copy()
    try:
        pyproject_data = read_toml()
        sources = pyproject_data.get("tool", {}).get("uv", {}).get("sources", {})

        for source_config in sources.values():
            if isinstance(source_config, dict):
                index = source_config.get("index")
                if index:
                    index_env = build_env_with_tool_repository_credentials(index)
                    env.update(index_env)
    except (FileNotFoundError, KeyError) as e:
        raise SystemExit(
            "Error. A valid pyproject.toml file is required. Check that a valid pyproject.toml file exists in the current directory."
        ) from e
    except Exception as e:
        raise SystemExit(f"Error: {e}") from e

    try:
        subprocess.run(  # noqa: S603
            ["uv", *uv_args],  # noqa: S607
            capture_output=False,
            env=env,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.secho(f"uv command failed with exit code {e.returncode}", fg="red")
        raise SystemExit(e.returncode) from e


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
    "-akn", "--agent-knowledge", is_flag=True, help="Reset AGENT KNOWLEDGE storage"
)
@click.option(
    "-k", "--kickoff-outputs", is_flag=True, help="Reset LATEST KICKOFF TASK OUTPUTS"
)
@click.option("-a", "--all", is_flag=True, help="Reset ALL memories")
def reset_memories(
    long: bool,
    short: bool,
    entities: bool,
    knowledge: bool,
    kickoff_outputs: bool,
    agent_knowledge: bool,
    all: bool,
) -> None:
    """
    Reset the crew memories (long, short, entity, latest_crew_kickoff_ouputs, knowledge, agent_knowledge). This will delete all the data saved.
    """
    try:
        memory_types = [
            long,
            short,
            entities,
            knowledge,
            agent_knowledge,
            kickoff_outputs,
            all,
        ]
        if not any(memory_types):
            click.echo(
                "Please specify at least one memory type to reset using the appropriate flags."
            )
            return
        reset_memories_command(
            long, short, entities, knowledge, agent_knowledge, kickoff_outputs, all
        )
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
    run_crew()


@crewai.command()
def update():
    """Update the pyproject.toml of the Crew project to use uv."""
    update_crew()


@crewai.command()
def login():
    """Sign Up/Login to CrewAI AMP."""
    Settings().clear_user_settings()
    AuthenticationCommand().login()


# DEPLOY CREWAI+ COMMANDS
@crewai.group()
def deploy():
    """Deploy the Crew CLI group."""


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
def deploy_push(uuid: str | None):
    """Deploy the Crew."""
    deploy_cmd = DeployCommand()
    deploy_cmd.deploy(uuid=uuid)


@deploy.command(name="status")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deply_status(uuid: str | None):
    """Get the status of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_status(uuid=uuid)


@deploy.command(name="logs")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_logs(uuid: str | None):
    """Get the logs of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_logs(uuid=uuid)


@deploy.command(name="remove")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_remove(uuid: str | None):
    """Remove a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.remove_crew(uuid=uuid)


@crewai.group()
def tool():
    """Tool Repository related commands."""


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


@crewai.group()
def triggers():
    """Trigger related commands. Use 'crewai triggers list' to see available triggers, or 'crewai triggers run app_slug/trigger_slug' to execute."""


@triggers.command(name="list")
def triggers_list():
    """List all available triggers from integrations."""
    triggers_cmd = TriggersCommand()
    triggers_cmd.list_triggers()


@triggers.command(name="run")
@click.argument("trigger_path")
def triggers_run(trigger_path: str):
    """Execute crew with trigger payload. Format: app_slug/trigger_slug"""
    triggers_cmd = TriggersCommand()
    triggers_cmd.execute_with_trigger(trigger_path)


@crewai.command()
def chat():
    """
    Start a conversation with the Crew, collecting user-supplied inputs,
    and using the Chat LLM to generate responses.
    """
    click.secho(
        "\nStarting a conversation with the Crew\nType 'exit' or Ctrl+C to quit.\n",
    )

    run_chat()


@crewai.group(invoke_without_command=True)
def org():
    """Organization management commands."""


@org.command("list")
def org_list():
    """List available organizations."""
    org_command = OrganizationCommand()
    org_command.list()


@org.command()
@click.argument("id")
def switch(id):
    """Switch to a specific organization."""
    org_command = OrganizationCommand()
    org_command.switch(id)


@org.command()
def current():
    """Show current organization when 'crewai org' is called without subcommands."""
    org_command = OrganizationCommand()
    org_command.current()


@crewai.group()
def enterprise():
    """Enterprise Configuration commands."""


@enterprise.command("configure")
@click.argument("enterprise_url")
def enterprise_configure(enterprise_url: str):
    """Configure CrewAI AMP OAuth2 settings from the provided Enterprise URL."""
    enterprise_command = EnterpriseConfigureCommand()
    enterprise_command.configure(enterprise_url)


@crewai.group()
def config():
    """CLI Configuration commands."""


@config.command("list")
def config_list():
    """List all CLI configuration parameters."""
    config_command = SettingsCommand()
    config_command.list()


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str):
    """Set a CLI configuration parameter."""
    config_command = SettingsCommand()
    config_command.set(key, value)


@config.command("reset")
def config_reset():
    """Reset all CLI configuration parameters to default values."""
    config_command = SettingsCommand()
    config_command.reset_all_settings()


@crewai.group()
def env():
    """Environment variable commands."""


@env.command("view")
def env_view():
    """View tracing-related environment variables."""
    import os
    from pathlib import Path

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Check for .env file
    env_file = Path(".env")
    env_file_exists = env_file.exists()

    # Create table for environment variables
    table = Table(show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Environment Variable", style="cyan", width=30)
    table.add_column("Value", style="white", width=20)
    table.add_column("Source", style="yellow", width=20)

    # Check CREWAI_TRACING_ENABLED
    crewai_tracing = os.getenv("CREWAI_TRACING_ENABLED", "")
    if crewai_tracing:
        table.add_row(
            "CREWAI_TRACING_ENABLED",
            crewai_tracing,
            "Environment/Shell",
        )
    else:
        table.add_row(
            "CREWAI_TRACING_ENABLED",
            "[dim]Not set[/dim]",
            "[dim]—[/dim]",
        )

    # Check other related env vars
    crewai_testing = os.getenv("CREWAI_TESTING", "")
    if crewai_testing:
        table.add_row("CREWAI_TESTING", crewai_testing, "Environment/Shell")

    crewai_user_id = os.getenv("CREWAI_USER_ID", "")
    if crewai_user_id:
        table.add_row("CREWAI_USER_ID", crewai_user_id, "Environment/Shell")

    crewai_org_id = os.getenv("CREWAI_ORG_ID", "")
    if crewai_org_id:
        table.add_row("CREWAI_ORG_ID", crewai_org_id, "Environment/Shell")

    # Check if .env file exists
    table.add_row(
        ".env file",
        "✅ Found" if env_file_exists else "❌ Not found",
        str(env_file.resolve()) if env_file_exists else "N/A",
    )

    panel = Panel(
        table,
        title="Tracing Environment Variables",
        border_style="blue",
        padding=(1, 2),
    )
    console.print("\n")
    console.print(panel)

    # Show helpful message
    if env_file_exists:
        console.print(
            "\n[dim]💡 Tip: To enable tracing via .env, add: CREWAI_TRACING_ENABLED=true[/dim]"
        )
    else:
        console.print(
            "\n[dim]💡 Tip: Create a .env file in your project root and add: CREWAI_TRACING_ENABLED=true[/dim]"
        )
    console.print()


@crewai.group()
def traces():
    """Trace collection management commands."""


@traces.command("enable")
def traces_enable():
    """Enable trace collection for crew/flow executions."""
    from rich.console import Console
    from rich.panel import Panel

    from crewai.events.listeners.tracing.utils import (
        _load_user_data,
        _save_user_data,
    )

    console = Console()

    # Update user data to enable traces
    user_data = _load_user_data()
    user_data["trace_consent"] = True
    user_data["first_execution_done"] = True
    _save_user_data(user_data)

    panel = Panel(
        "✅ Trace collection has been enabled!\n\n"
        "Your crew/flow executions will now send traces to CrewAI+.\n"
        "Use 'crewai traces disable' to turn off trace collection.",
        title="Traces Enabled",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


@traces.command("disable")
def traces_disable():
    """Disable trace collection for crew/flow executions."""
    from rich.console import Console
    from rich.panel import Panel

    from crewai.events.listeners.tracing.utils import (
        _load_user_data,
        _save_user_data,
    )

    console = Console()

    # Update user data to disable traces
    user_data = _load_user_data()
    user_data["trace_consent"] = False
    user_data["first_execution_done"] = True
    _save_user_data(user_data)

    panel = Panel(
        "❌ Trace collection has been disabled!\n\n"
        "Your crew/flow executions will no longer send traces.\n"
        "Use 'crewai traces enable' to turn trace collection back on.",
        title="Traces Disabled",
        border_style="red",
        padding=(1, 2),
    )
    console.print(panel)


@traces.command("status")
def traces_status():
    """Show current trace collection status."""
    import os

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    from crewai.events.listeners.tracing.utils import (
        _load_user_data,
        is_tracing_enabled,
    )

    console = Console()
    user_data = _load_user_data()

    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    # Check environment variable
    env_enabled = os.getenv("CREWAI_TRACING_ENABLED", "false")
    table.add_row("CREWAI_TRACING_ENABLED", env_enabled)

    # Check user consent
    trace_consent = user_data.get("trace_consent")
    if trace_consent is True:
        consent_status = "✅ Enabled (user consented)"
    elif trace_consent is False:
        consent_status = "❌ Disabled (user declined)"
    else:
        consent_status = "⚪ Not set (first-time user)"
    table.add_row("User Consent", consent_status)

    # Check overall status
    if is_tracing_enabled():
        overall_status = "✅ ENABLED"
        border_style = "green"
    else:
        overall_status = "❌ DISABLED"
        border_style = "red"
    table.add_row("Overall Status", overall_status)

    panel = Panel(
        table,
        title="Trace Collection Status",
        border_style=border_style,
        padding=(1, 2),
    )
    console.print(panel)


if __name__ == "__main__":
    crewai()
