from pathlib import Path
import sys

import click
from crewai_core.telemetry import Telemetry

from crewai_cli.constants import crewai_extras_for_model, provider_for_model
from crewai_cli.git import initialize_if_git_available
from crewai_cli.input_prompt import is_interactive
from crewai_cli.utils import (
    copy_assistant_imports,
    copy_assistant_instructions,
    enable_prompt_line_editing,
    get_or_create_project_id,
    missing_provider_keys,
    write_env_file,
)
from crewai_cli.version import get_crewai_tools_dependency


DECLARATIVE_FLOW_FOLDERS = ("crews", "tools", "knowledge", "skills")


def create_flow(
    name: str,
    *,
    declarative: bool = False,
    provider: str | None = None,
    skip_provider: bool = False,
) -> Path | None:
    """Create a new flow.

    At a terminal, a Python flow asks for a model and its API keys and saves
    them to ``.env``. A declarative flow, whose starter calls no model, never
    asks or names a key, and a run without a terminal never asks.

    Args:
        name: The flow's name.
        declarative: Create a declarative flow instead of a Python flow.
        provider: A provider, such as ``anthropic``, whose first listed model
            the flow uses, or a ``provider/model`` string.
        skip_provider: Ask for no model or API keys.

    Returns:
        The flow's folder, or None when that folder already exists.

    Raises:
        click.BadParameter: ``provider`` is neither a provider with a listed
            model nor a ``provider/model`` string.
    """
    from crewai_cli.create_json_crew import (
        _PROVIDER_MODELS,
        _default_model_for_provider,
        _select_model,
        _setup_env,
    )

    model = _default_model_for_provider(provider)
    if provider and model is None:
        raise click.BadParameter(
            f"{provider!r} has no default model. Pass one of "
            f"{', '.join(_PROVIDER_MODELS)}, or a provider/model string.",
            param_hint="'--provider'",
        )

    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    click.secho(f"Creating flow {folder_name}...", fg="green", bold=True)

    project_root = Path(folder_name)
    if project_root.exists():
        click.secho(f"Error: Folder {folder_name} already exists.", fg="red")
        return None

    ask = (
        not declarative
        and not skip_provider
        and is_interactive()
        and sys.stdout.isatty()
    )
    if ask:
        enable_prompt_line_editing()
        if model is None:
            model = _select_model()

    telemetry = Telemetry()
    telemetry.flow_creation_span(class_name)

    crewai_dependency = get_crewai_tools_dependency(
        extras=crewai_extras_for_model(model) if model else ()
    )
    if declarative:
        _create_declarative_flow(
            name, class_name, folder_name, project_root, crewai_dependency
        )
    else:
        _create_python_flow(
            name, class_name, folder_name, project_root, crewai_dependency
        )

    if model is not None:
        write_env_file(project_root, {"MODEL": model})

    # Minted at creation so the project has a stable identity from run one.
    project_id = get_or_create_project_id(project_root / "pyproject.toml")
    # After the mint, unlike flow_creation_span above, which fires before the id exists.
    telemetry.project_created_span("flow", project_id)
    initialize_if_git_available(project_root)

    # The declarative starter calls no model, so it needs no key.
    keys_to_set: list[str] = []
    if ask and model is not None:
        # Last, so Ctrl-C at a key prompt leaves a complete project with MODEL saved.
        _setup_env(project_root, [model])
    elif not declarative:
        if model is None:
            # With no MODEL, crewAI runs the flow on its default OpenAI model.
            keys_to_set = missing_provider_keys("openai", {})
        else:
            keys_to_set = missing_provider_keys(
                provider_for_model(model) or model.split("/")[0], {}
            )

    click.secho(f"Flow {name} created successfully!", fg="green", bold=True)
    click.echo()
    click.secho("  Next steps:", bold=True)
    click.echo()
    click.echo(f"    cd {folder_name}")
    click.echo()
    if keys_to_set:
        click.secho(
            f"  Set {', '.join(keys_to_set)} in .env before `crewai run`", fg="cyan"
        )
        click.echo()
    click.secho("  Run your flow:", fg="cyan")
    click.echo("    crewai run")
    click.echo()
    return project_root


def _create_python_flow(
    name: str,
    class_name: str,
    folder_name: str,
    project_root: Path,
    crewai_dependency: str,
) -> None:
    (project_root / "src" / folder_name).mkdir(parents=True)
    (project_root / "src" / folder_name / "crews").mkdir(parents=True)
    (project_root / "src" / folder_name / "tools").mkdir(parents=True)
    (project_root / "tests").mkdir(exist_ok=True)

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "flow"

    copy_assistant_instructions(project_root)

    root_template_files = [".gitignore", "pyproject.toml", "README.md"]
    src_template_files = ["__init__.py", "main.py"]
    tools_template_files = ["tools/__init__.py", "tools/custom_tool.py"]

    crew_folders = [
        "content_crew",
    ]

    def process_file(src_file: Path, dst_file: Path) -> None:
        if src_file.suffix in [".pyc", ".pyo", ".pyd"]:
            return

        try:
            with open(src_file, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as e:
            click.secho(f"Error processing file {src_file}: {e}", fg="red")
            return

        content = content.replace("{{name}}", name)
        content = content.replace("{{flow_name}}", class_name)
        content = content.replace("{{folder_name}}", folder_name)
        content = content.replace("{{crewai_tools_dependency}}", crewai_dependency)

        with open(dst_file, "w") as file:
            file.write(content)

    for file_name in root_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / file_name
        process_file(src_file, dst_file)

    for file_name in src_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    for file_name in tools_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    for crew_folder in crew_folders:
        src_crew_folder = templates_dir / "crews" / crew_folder
        dst_crew_folder = project_root / "src" / folder_name / "crews" / crew_folder
        if src_crew_folder.exists():
            for src_file in src_crew_folder.rglob("*"):
                if src_file.is_file():
                    relative_path = src_file.relative_to(src_crew_folder)
                    dst_file = dst_crew_folder / relative_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    process_file(src_file, dst_file)
        else:
            click.secho(
                f"Warning: Crew folder {crew_folder} not found in template.",
                fg="yellow",
            )


def _create_declarative_flow(
    name: str,
    class_name: str,
    folder_name: str,
    project_root: Path,
    crewai_dependency: str,
) -> None:
    project_root.mkdir(parents=True)
    package_root = project_root / "src" / folder_name
    package_root.mkdir(parents=True)
    for folder in DECLARATIVE_FLOW_FOLDERS:
        (package_root / folder).mkdir()

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "declarative_flow"
    root_template_files = {".gitignore", "AGENTS.md", "README.md", "pyproject.toml"}

    for src_file in templates_dir.rglob("*"):
        if not src_file.is_file():
            continue

        relative_path = src_file.relative_to(templates_dir)
        dst_file = (
            project_root / relative_path
            if relative_path.name in root_template_files
            else package_root / relative_path
        )
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        content = src_file.read_text(encoding="utf-8")
        content = content.replace("{{name}}", name)
        content = content.replace("{{flow_name}}", class_name)
        content = content.replace("{{folder_name}}", folder_name)
        content = content.replace("{{crewai_tools_dependency}}", crewai_dependency)
        dst_file.write_text(content, encoding="utf-8")

    copy_assistant_imports(project_root)
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    for folder in DECLARATIVE_FLOW_FOLDERS:
        (package_root / folder / ".gitkeep").write_text("", encoding="utf-8")
