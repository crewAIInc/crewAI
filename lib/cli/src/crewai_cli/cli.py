from __future__ import annotations

from importlib.metadata import version as get_version
import os
import subprocess
from typing import Any

import click
from crewai_core.token_manager import TokenManager

from crewai_cli.add_crew_to_flow import add_crew_to_flow
from crewai_cli.authentication.main import AuthenticationCommand
from crewai_cli.config import Settings
from crewai_cli.create_crew import create_crew
from crewai_cli.create_flow import create_flow
from crewai_cli.crew_chat import run_chat
from crewai_cli.deploy.main import DeployCommand
from crewai_cli.enterprise.main import EnterpriseConfigureCommand
from crewai_cli.evaluate_crew import evaluate_crew
from crewai_cli.install_crew import install_crew
from crewai_cli.kickoff_flow import kickoff_flow
from crewai_cli.organization.main import OrganizationCommand
from crewai_cli.plot_flow import plot_flow
from crewai_cli.remote_template.main import TemplateCommand
from crewai_cli.replay_from_task import replay_task_command
from crewai_cli.reset_memories_command import reset_memories_command
from crewai_cli.run_crew import run_crew
from crewai_cli.settings.main import SettingsCommand
from crewai_cli.task_outputs import load_task_outputs
from crewai_cli.tools.main import ToolCommand
from crewai_cli.train_crew import train_crew
from crewai_cli.triggers.main import TriggersCommand
from crewai_cli.update_crew import update_crew
from crewai_cli.user_data import (
    _load_user_data,
    is_tracing_enabled,
    update_user_data,
)
from crewai_cli.utils import build_env_with_all_tool_credentials, read_toml


def _get_cli_version() -> str:
    """Return the best available version string for the CLI."""
    # Prefer crewai version if installed (keeps existing UX)
    try:
        return get_version("crewai")
    except Exception:  # noqa: S110
        pass
    try:
        return get_version("crewai-cli")
    except Exception:
        return "unknown"


@click.group()
@click.version_option(_get_cli_version())
def crewai() -> None:
    """Top-level command group for crewai."""


@crewai.command(
    name="uv",
    context_settings={"ignore_unknown_options": True},
)
@click.argument("uv_args", nargs=-1, type=click.UNPROCESSED)
def uv(uv_args: tuple[str, ...]) -> None:
    """A wrapper around uv commands that adds custom tool authentication through env vars."""
    try:
        # Verify pyproject.toml exists first
        read_toml()
    except FileNotFoundError as e:
        raise SystemExit(
            "Error. A valid pyproject.toml file is required. Check that a valid pyproject.toml file exists in the current directory."
        ) from e
    except Exception as e:
        raise SystemExit(f"Error: {e}") from e

    env = build_env_with_all_tool_credentials()

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
def create(
    type: str, name: str, provider: str | None, skip_provider: bool = False
) -> None:
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
def version(tools: bool) -> None:
    """Show the installed version of crewai."""
    try:
        crewai_version = get_version("crewai")
    except Exception:
        crewai_version = "unknown version"
    click.echo(f"crewai version: {crewai_version}")

    if tools:
        try:
            tools_version = get_version("crewai-tools")
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
def train(n_iterations: int, filename: str) -> None:
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
@click.option(
    "-f",
    "--filename",
    "trained_agents_file",
    type=str,
    default=None,
    help=(
        "Path to a trained-agents pickle (produced by `crewai train -f`). "
        "When set, agents load suggestions from this file instead of the "
        "default trained_agents_data.pkl. Equivalent to setting "
        "CREWAI_TRAINED_AGENTS_FILE."
    ),
)
def replay(task_id: str, trained_agents_file: str | None) -> None:
    """Replay the crew execution from a specific task.

    Args:
        task_id: The ID of the task to replay from.
        trained_agents_file: Optional trained-agents pickle path.
    """
    try:
        click.echo(f"Replaying the crew from task {task_id}")
        replay_task_command(task_id, trained_agents_file=trained_agents_file)
    except Exception as e:
        click.echo(f"An error occurred while replaying: {e}", err=True)


@crewai.command()
def log_tasks_outputs() -> None:
    """Retrieve your latest crew.kickoff() task outputs."""
    try:
        tasks = load_task_outputs()

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
@click.option("-m", "--memory", is_flag=True, help="Reset MEMORY")
@click.option(
    "-l",
    "--long",
    is_flag=True,
    hidden=True,
    help="[Deprecated: use --memory] Reset memory",
)
@click.option(
    "-s",
    "--short",
    is_flag=True,
    hidden=True,
    help="[Deprecated: use --memory] Reset memory",
)
@click.option(
    "-e",
    "--entities",
    is_flag=True,
    hidden=True,
    help="[Deprecated: use --memory] Reset memory",
)
@click.option("-kn", "--knowledge", is_flag=True, help="Reset KNOWLEDGE storage")
@click.option(
    "-akn", "--agent-knowledge", is_flag=True, help="Reset AGENT KNOWLEDGE storage"
)
@click.option(
    "-k", "--kickoff-outputs", is_flag=True, help="Reset LATEST KICKOFF TASK OUTPUTS"
)
@click.option("-a", "--all", is_flag=True, help="Reset ALL memories")
def reset_memories(
    memory: bool,
    long: bool,
    short: bool,
    entities: bool,
    knowledge: bool,
    kickoff_outputs: bool,
    agent_knowledge: bool,
    all: bool,
) -> None:
    """Reset the crew memories (memory, knowledge, agent_knowledge, kickoff_outputs). This will delete all the data saved."""
    try:
        if long or short or entities:
            legacy_used = [
                f
                for f, v in [
                    ("--long", long),
                    ("--short", short),
                    ("--entities", entities),
                ]
                if v
            ]
            click.echo(
                f"Warning: {', '.join(legacy_used)} {'is' if len(legacy_used) == 1 else 'are'} "
                "deprecated. Use --memory (-m) instead. All memory is now unified."
            )
            memory = True

        memory_types = [
            memory,
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
        reset_memories_command(memory, knowledge, agent_knowledge, kickoff_outputs, all)
    except Exception as e:
        click.echo(f"An error occurred while resetting memories: {e}", err=True)


@crewai.command()
@click.option(
    "--storage-path",
    type=str,
    default=None,
    help="Path to LanceDB memory directory. If omitted, uses ./.crewai/memory.",
)
@click.option(
    "--embedder-provider",
    type=str,
    default=None,
    help="Embedder provider for recall queries (e.g. openai, google-vertex, cohere, ollama).",
)
@click.option(
    "--embedder-model",
    type=str,
    default=None,
    help="Embedder model name (e.g. text-embedding-3-small, gemini-embedding-001).",
)
@click.option(
    "--embedder-config",
    type=str,
    default=None,
    help='Full embedder config as JSON (e.g. \'{"provider": "cohere", "config": {"model_name": "embed-v4.0"}}\').',
)
def memory(
    storage_path: str | None,
    embedder_provider: str | None,
    embedder_model: str | None,
    embedder_config: str | None,
) -> None:
    """Open the Memory TUI to browse scopes and recall memories."""
    try:
        from crewai_cli.memory_tui import MemoryTUI
    except ImportError as exc:
        click.echo(
            "Textual is required for the memory TUI but could not be imported. "
            "Try reinstalling crewai or: pip install textual"
        )
        raise SystemExit(1) from exc

    # Build embedder spec from CLI flags.
    embedder_spec: dict[str, Any] | None = None
    if embedder_config:
        import json as _json

        try:
            embedder_spec = _json.loads(embedder_config)
        except _json.JSONDecodeError as exc:
            click.echo(f"Invalid --embedder-config JSON: {exc}")
            raise SystemExit(1) from exc
    elif embedder_provider:
        cfg: dict[str, str] = {}
        if embedder_model:
            cfg["model_name"] = embedder_model
        embedder_spec = {"provider": embedder_provider, "config": cfg}

    app = MemoryTUI(storage_path=storage_path, embedder_config=embedder_spec)
    app.run()


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
@click.option(
    "-f",
    "--filename",
    "trained_agents_file",
    type=str,
    default=None,
    help=(
        "Path to a trained-agents pickle (produced by `crewai train -f`). "
        "When set, agents load suggestions from this file instead of the "
        "default trained_agents_data.pkl. Equivalent to setting "
        "CREWAI_TRAINED_AGENTS_FILE."
    ),
)
def test(n_iterations: int, model: str, trained_agents_file: str | None) -> None:
    """Test the crew and evaluate the results."""
    click.echo(f"Testing the crew for {n_iterations} iterations with model {model}")
    evaluate_crew(n_iterations, model, trained_agents_file=trained_agents_file)


@crewai.command(
    context_settings={
        "ignore_unknown_options": True,
        "allow_extra_args": True,
    }
)
@click.pass_context
def install(context: click.Context) -> None:
    """Install the Crew."""
    install_crew(context.args)


@crewai.command()
@click.option(
    "-f",
    "--filename",
    "trained_agents_file",
    type=str,
    default=None,
    help=(
        "Path to a trained-agents pickle (produced by `crewai train -f`). "
        "When set, agents load suggestions from this file instead of the "
        "default trained_agents_data.pkl. Equivalent to setting "
        "CREWAI_TRAINED_AGENTS_FILE."
    ),
)
def run(trained_agents_file: str | None) -> None:
    """Run the Crew."""
    run_crew(trained_agents_file=trained_agents_file)


@crewai.command()
def update() -> None:
    """Update the pyproject.toml of the Crew project to use uv."""
    update_crew()


@crewai.command()
def login() -> None:
    """Sign Up/Login to CrewAI AMP."""
    Settings().clear_user_settings()
    AuthenticationCommand().login()


@crewai.command()
@click.option(
    "--reset", is_flag=True, help="Also reset all CLI configuration to defaults"
)
def logout(reset: bool) -> None:
    """Logout from CrewAI AMP."""
    settings = Settings()
    if reset:
        settings.reset()
        click.echo("Successfully logged out and reset all CLI configuration.")
    else:
        TokenManager().clear_tokens()
        settings.clear_user_settings()
        click.echo("Successfully logged out from CrewAI AMP.")


# DEPLOY CREWAI+ COMMANDS
@crewai.group()
def deploy() -> None:
    """Deploy the Crew CLI group."""


@deploy.command(name="create")
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
@click.option(
    "--skip-validate",
    is_flag=True,
    help="Skip the pre-deploy validation checks.",
)
def deploy_create(yes: bool, skip_validate: bool) -> None:
    """Create a Crew deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.create_crew(yes, skip_validate=skip_validate)


@deploy.command(name="list")
def deploy_list() -> None:
    """List all deployments."""
    deploy_cmd = DeployCommand()
    deploy_cmd.list_crews()


@deploy.command(name="push")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
@click.option(
    "--skip-validate",
    is_flag=True,
    help="Skip the pre-deploy validation checks.",
)
def deploy_push(uuid: str | None, skip_validate: bool) -> None:
    """Deploy the Crew."""
    deploy_cmd = DeployCommand()
    deploy_cmd.deploy(uuid=uuid, skip_validate=skip_validate)


@deploy.command(name="validate")
def deploy_validate() -> None:
    """Validate the current project against common deployment failures.

    Runs the same pre-deploy checks that `crewai deploy create` and
    `crewai deploy push` run automatically, without contacting the platform.
    Exits non-zero if any blocking issues are found.
    """
    from crewai_cli.deploy.validate import run_validate_command

    run_validate_command()


@deploy.command(name="status")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deply_status(uuid: str | None) -> None:
    """Get the status of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_status(uuid=uuid)


@deploy.command(name="logs")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_logs(uuid: str | None) -> None:
    """Get the logs of a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.get_crew_logs(uuid=uuid)


@deploy.command(name="remove")
@click.option("-u", "--uuid", type=str, help="Crew UUID parameter")
def deploy_remove(uuid: str | None) -> None:
    """Remove a deployment."""
    deploy_cmd = DeployCommand()
    deploy_cmd.remove_crew(uuid=uuid)


@crewai.group()
def tool() -> None:
    """Tool Repository related commands."""


@tool.command(name="create")
@click.argument("handle")
def tool_create(handle: str) -> None:
    tool_cmd = ToolCommand()
    tool_cmd.create(handle)


@tool.command(name="install")
@click.argument("handle")
def tool_install(handle: str) -> None:
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
def tool_publish(is_public: bool, force: bool) -> None:
    tool_cmd = ToolCommand()
    tool_cmd.login()
    tool_cmd.publish(is_public, force)


@crewai.group()
def template() -> None:
    """Browse and install project templates."""


@template.command(name="list")
def template_list() -> None:
    """List available templates and select one to install."""
    template_cmd = TemplateCommand()
    template_cmd.list_templates()


@template.command(name="add")
@click.argument("name")
@click.option(
    "-o",
    "--output-dir",
    type=str,
    default=None,
    help="Directory name for the template (defaults to template name)",
)
def template_add(name: str, output_dir: str | None) -> None:
    """Add a template to the current directory."""
    template_cmd = TemplateCommand()
    template_cmd.add_template(name, output_dir)


@crewai.group()
def flow() -> None:
    """Flow related commands."""


@flow.command(name="kickoff")
def flow_run() -> None:
    """Kickoff the Flow."""
    click.echo("Running the Flow")
    kickoff_flow()


@flow.command(name="plot")
def flow_plot() -> None:
    """Plot the Flow."""
    click.echo("Plotting the Flow")
    plot_flow()


@flow.command(name="add-crew")
@click.argument("crew_name")
def flow_add_crew(crew_name: str) -> None:
    """Add a crew to an existing flow."""
    click.echo(f"Adding crew {crew_name} to the flow")
    add_crew_to_flow(crew_name)


@crewai.group()
def triggers() -> None:
    """Trigger related commands. Use 'crewai triggers list' to see available triggers, or 'crewai triggers run app_slug/trigger_slug' to execute."""


@triggers.command(name="list")
def triggers_list() -> None:
    """List all available triggers from integrations."""
    triggers_cmd = TriggersCommand()
    triggers_cmd.list_triggers()


@triggers.command(name="run")
@click.argument("trigger_path")
def triggers_run(trigger_path: str) -> None:
    """Execute crew with trigger payload. Format: app_slug/trigger_slug"""
    triggers_cmd = TriggersCommand()
    triggers_cmd.execute_with_trigger(trigger_path)


@crewai.command()
def chat() -> None:
    """Start a conversation with the Crew, collecting user-supplied inputs,
    and using the Chat LLM to generate responses.
    """
    click.secho(
        "\nStarting a conversation with the Crew\nType 'exit' or Ctrl+C to quit.\n",
    )
    run_chat()


@crewai.group(invoke_without_command=True)
def org() -> None:
    """Organization management commands."""


@org.command("list")
def org_list() -> None:
    """List available organizations."""
    org_command = OrganizationCommand()
    org_command.list()


@org.command()
@click.argument("id")
def switch(id: str) -> None:
    """Switch to a specific organization."""
    org_command = OrganizationCommand()
    org_command.switch(id)


@org.command()
def current() -> None:
    """Show current organization when 'crewai org' is called without subcommands."""
    org_command = OrganizationCommand()
    org_command.current()


@crewai.group()
def enterprise() -> None:
    """Enterprise Configuration commands."""


@enterprise.command("configure")
@click.argument("enterprise_url")
def enterprise_configure(enterprise_url: str) -> None:
    """Configure CrewAI AMP OAuth2 settings from the provided Enterprise URL."""
    enterprise_command = EnterpriseConfigureCommand()
    enterprise_command.configure(enterprise_url)


@crewai.group()
def config() -> None:
    """CLI Configuration commands."""


@config.command("list")
def config_list() -> None:
    """List all CLI configuration parameters."""
    config_command = SettingsCommand()
    config_command.list()


@config.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key: str, value: str) -> None:
    """Set a CLI configuration parameter."""
    config_command = SettingsCommand()
    config_command.set(key, value)


@config.command("reset")
def config_reset() -> None:
    """Reset all CLI configuration parameters to default values."""
    config_command = SettingsCommand()
    config_command.reset_all_settings()


@crewai.group()
def env() -> None:
    """Environment variable commands."""


@env.command("view")
def env_view() -> None:
    """View tracing-related environment variables."""
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
def traces() -> None:
    """Trace collection management commands."""


@traces.command("enable")
def traces_enable() -> None:
    """Enable trace collection for crew/flow executions."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    update_user_data({"trace_consent": True, "first_execution_done": True})

    panel = Panel(
        "✅ Trace collection enabled.\n\n"
        "Your crew/flow executions will now send traces to CrewAI+.\n"
        "Use 'crewai traces disable' to opt out.",
        title="Traces Enabled",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


@traces.command("disable")
def traces_disable() -> None:
    """Disable trace collection for crew/flow executions."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()

    update_user_data({"trace_consent": False, "first_execution_done": True})

    panel = Panel(
        "❌ Trace collection disabled.\n\n"
        "Your crew/flow executions will no longer send traces "
        "(unless [bold]CREWAI_TRACING_ENABLED=true[/bold] is set in the environment, "
        "which overrides the opt-out).\n"
        "Use 'crewai traces enable' to opt back in.",
        title="Traces Disabled",
        border_style="red",
        padding=(1, 2),
    )
    console.print(panel)


@traces.command("status")
def traces_status() -> None:
    """Show current trace collection status."""

    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

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


@crewai.group(invoke_without_command=True)
@click.option(
    "--location", default="./.checkpoints", help="Checkpoint directory or SQLite file."
)
@click.pass_context
def checkpoint(ctx: click.Context, location: str) -> None:
    """Browse and inspect checkpoints. Launches a TUI when called without a subcommand."""
    from crewai_cli.checkpoint_cli import _detect_location

    location = _detect_location(location)
    ctx.ensure_object(dict)
    ctx.obj["location"] = location
    if ctx.invoked_subcommand is None:
        from crewai_cli.checkpoint_tui import run_checkpoint_tui

        run_checkpoint_tui(location)


@checkpoint.command("list")
@click.argument("location", default="./.checkpoints")
def checkpoint_list(location: str) -> None:
    """List checkpoints in a directory."""
    from crewai_cli.checkpoint_cli import _detect_location, list_checkpoints

    list_checkpoints(_detect_location(location))


@checkpoint.command("info")
@click.argument("path", default="./.checkpoints")
def checkpoint_info(path: str) -> None:
    """Show details of a checkpoint. Pass a file or directory for latest."""
    from crewai_cli.checkpoint_cli import _detect_location, info_checkpoint

    info_checkpoint(_detect_location(path))


@checkpoint.command("resume")
@click.argument("checkpoint_id", required=False, default=None)
@click.pass_context
def checkpoint_resume(ctx: click.Context, checkpoint_id: str | None) -> None:
    """Resume from a checkpoint. Defaults to the most recent."""
    from crewai_cli.checkpoint_cli import resume_checkpoint

    resume_checkpoint(ctx.obj["location"], checkpoint_id)


@checkpoint.command("diff")
@click.argument("id1")
@click.argument("id2")
@click.pass_context
def checkpoint_diff(ctx: click.Context, id1: str, id2: str) -> None:
    """Compare two checkpoints side-by-side."""
    from crewai_cli.checkpoint_cli import diff_checkpoints

    diff_checkpoints(ctx.obj["location"], id1, id2)


@checkpoint.command("prune")
@click.option(
    "--keep", type=int, default=None, help="Keep the N most recent checkpoints."
)
@click.option(
    "--older-than",
    default=None,
    help="Remove checkpoints older than duration (e.g. 7d, 24h, 30m).",
)
@click.option(
    "--dry-run", is_flag=True, help="Show what would be pruned without deleting."
)
@click.pass_context
def checkpoint_prune(
    ctx: click.Context, keep: int | None, older_than: str | None, dry_run: bool
) -> None:
    """Remove old checkpoints."""
    from crewai_cli.checkpoint_cli import prune_checkpoints

    prune_checkpoints(ctx.obj["location"], keep, older_than, dry_run)


if __name__ == "__main__":
    crewai()
