import sys
from pathlib import Path

import click

from crewai.cli.constants import ENV_VARS
from crewai.cli.provider import (
    PROVIDERS,
    get_provider_data,
    select_model,
    select_provider,
)
from crewai.cli.utils import copy_template, load_env_vars, write_env_file


def create_folder_structure(name, parent_folder=None):
    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    if parent_folder:
        folder_path = Path(parent_folder) / folder_name
    else:
        folder_path = Path(folder_name)

    if folder_path.exists():
        if not click.confirm(
            f"Folder {folder_name} already exists. Do you want to override it?"
        ):
            click.secho("Operation cancelled.", fg="yellow")
            sys.exit(0)
        click.secho(f"Overriding folder {folder_name}...", fg="green", bold=True)
    else:
        click.secho(
            f"Creating {'crew' if parent_folder else 'folder'} {folder_name}...",
            fg="green",
            bold=True,
        )

    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        (folder_path / "tests").mkdir(exist_ok=True)
        if not parent_folder:
            (folder_path / "src" / folder_name).mkdir(parents=True)
            (folder_path / "src" / folder_name / "tools").mkdir(parents=True)
            (folder_path / "src" / folder_name / "config").mkdir(parents=True)

    return folder_path, folder_name, class_name


def copy_template_files(folder_path, name, class_name, parent_folder):
    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "crew"

    root_template_files = (
        [".gitignore", "pyproject.toml", "README.md"] if not parent_folder else []
    )
    tools_template_files = ["tools/custom_tool.py", "tools/__init__.py"]
    config_template_files = ["config/agents.yaml", "config/tasks.yaml"]
    src_template_files = (
        ["__init__.py", "main.py", "crew.py"] if not parent_folder else ["crew.py"]
    )

    for file_name in root_template_files:
        src_file = templates_dir / file_name
        dst_file = folder_path / file_name
        copy_template(src_file, dst_file, name, class_name, folder_path.name)

    src_folder = (
        folder_path / "src" / folder_path.name if not parent_folder else folder_path
    )

    for file_name in src_template_files:
        src_file = templates_dir / file_name
        dst_file = src_folder / file_name
        copy_template(src_file, dst_file, name, class_name, folder_path.name)

    if not parent_folder:
        for file_name in tools_template_files + config_template_files:
            src_file = templates_dir / file_name
            dst_file = src_folder / file_name
            copy_template(src_file, dst_file, name, class_name, folder_path.name)


def create_crew(name, provider=None, skip_provider=False, parent_folder=None):
    folder_path, folder_name, class_name = create_folder_structure(name, parent_folder)
    env_vars = load_env_vars(folder_path)
    if not skip_provider:
        if not provider:
            provider_models = get_provider_data()
            if not provider_models:
                return

        existing_provider = None
        for provider, env_keys in ENV_VARS.items():
            if any(key in env_vars for key in env_keys):
                existing_provider = provider
                break

        if existing_provider:
            if not click.confirm(
                f"Found existing environment variable configuration for {existing_provider.capitalize()}. Do you want to override it?"
            ):
                click.secho("Keeping existing provider configuration.", fg="yellow")
                return

        provider_models = get_provider_data()
        if not provider_models:
            return

        while True:
            selected_provider = select_provider(provider_models)
            if selected_provider is None:  # User typed 'q'
                click.secho("Exiting...", fg="yellow")
                sys.exit(0)
            if selected_provider:  # Valid selection
                break
            click.secho(
                "No provider selected. Please try again or press 'q' to exit.", fg="red"
            )

        while True:
            selected_model = select_model(selected_provider, provider_models)
            if selected_model is None:  # User typed 'q'
                click.secho("Exiting...", fg="yellow")
                sys.exit(0)
            if selected_model:  # Valid selection
                break
            click.secho(
                "No model selected. Please try again or press 'q' to exit.", fg="red"
            )

        if selected_provider in PROVIDERS:
            api_key_var = ENV_VARS[selected_provider][0]
        else:
            api_key_var = click.prompt(
                f"Enter the environment variable name for your {selected_provider.capitalize()} API key",
                type=str,
                default="",
            )

        api_key_value = ""
        click.echo(
            f"Enter your {selected_provider.capitalize()} API key (press Enter to skip): ",
            nl=False,
        )
        try:
            api_key_value = input()
        except (KeyboardInterrupt, EOFError):
            api_key_value = ""

        if api_key_value.strip():
            env_vars = {api_key_var: api_key_value}
            write_env_file(folder_path, env_vars)
            click.secho("API key saved to .env file", fg="green")
        else:
            click.secho(
                "No API key provided. Skipping .env file creation.", fg="yellow"
            )

        env_vars["MODEL"] = selected_model
        click.secho(f"Selected model: {selected_model}", fg="green")

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "crew"

    root_template_files = (
        [".gitignore", "pyproject.toml", "README.md"] if not parent_folder else []
    )
    tools_template_files = ["tools/custom_tool.py", "tools/__init__.py"]
    config_template_files = ["config/agents.yaml", "config/tasks.yaml"]
    src_template_files = (
        ["__init__.py", "main.py", "crew.py"] if not parent_folder else ["crew.py"]
    )

    for file_name in root_template_files:
        src_file = templates_dir / file_name
        dst_file = folder_path / file_name
        copy_template(src_file, dst_file, name, class_name, folder_name)

    src_folder = folder_path / "src" / folder_name if not parent_folder else folder_path

    for file_name in src_template_files:
        src_file = templates_dir / file_name
        dst_file = src_folder / file_name
        copy_template(src_file, dst_file, name, class_name, folder_name)

    if not parent_folder:
        for file_name in tools_template_files + config_template_files:
            src_file = templates_dir / file_name
            dst_file = src_folder / file_name
            copy_template(src_file, dst_file, name, class_name, folder_name)

    click.secho(f"Crew {name} created successfully!", fg="green", bold=True)
