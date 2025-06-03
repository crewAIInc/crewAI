import shutil
import sys
from pathlib import Path

import click

from crewai.cli.constants import ENV_VARS, MODELS
from crewai.cli.provider import (
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
        shutil.rmtree(folder_path)  # Delete the existing folder and its contents

    click.secho(
        f"Creating {'crew' if parent_folder else 'folder'} {folder_name}...",
        fg="green",
        bold=True,
    )

    folder_path.mkdir(parents=True)
    (folder_path / "tests").mkdir(exist_ok=True)
    (folder_path / "knowledge").mkdir(exist_ok=True)
    if not parent_folder:
        (folder_path / "src" / folder_name).mkdir(parents=True)
        (folder_path / "src" / folder_name / "tools").mkdir(parents=True)
        (folder_path / "src" / folder_name / "config").mkdir(parents=True)

    return folder_path, folder_name, class_name


def copy_template_files(folder_path, name, class_name, parent_folder):
    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "crew"

    root_template_files = (
        [
            ".gitignore",
            "pyproject.toml",
            "README.md",
            "knowledge/user_preference.txt",
        ]
        if not parent_folder
        else []
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
        provider_models = get_provider_data()
        if not provider_models:
            click.secho("Could not retrieve provider data.", fg="red")
            return

        selected_provider = None

        if provider: 
            provider = provider.lower()
            if provider in provider_models:
                selected_provider = provider
                click.secho(f"Using specified provider: {selected_provider.capitalize()}", fg="green")
            else:
                click.secho(f"Warning: Specified provider '{provider}' is not recognized. Please select one.", fg="yellow")

        if not selected_provider:
            existing_provider = None
            for p, env_keys in ENV_VARS.items():
                if any(
                    "key_name" in details and details["key_name"] in env_vars
                    for details in env_keys
                ):
                    existing_provider = p
                    break

            if existing_provider:
                if not click.confirm(
                    f"Found existing environment variable configuration for {existing_provider.capitalize()}. Do you want to override it?"
                ):
                    click.secho("Keeping existing provider configuration. Exiting provider setup.", fg="yellow")
                    copy_template_files(folder_path, name, class_name, parent_folder)
                    click.secho(f"Crew '{name}' created successfully!", fg="green")
                    click.secho(f"To run your crew, cd into '{folder_name}' and run 'crewai run'", fg="cyan")
                    return
                else:
                    pass

            while True:
                selected_provider = select_provider(provider_models)
                if selected_provider is None:  
                    click.secho("Exiting...", fg="yellow")
                    sys.exit(0)
                if selected_provider:  
                    break
                click.secho(
                    "No provider selected. Please try again or press 'q' to exit.", fg="red"
                )

        if not selected_provider:
             click.secho("Provider selection failed. Exiting.", fg="red")
             sys.exit(1)


        if selected_provider in MODELS and MODELS[selected_provider]:
            while True:
                selected_model = select_model(selected_provider, provider_models)
                if selected_model is None:  
                    click.secho("Exiting...", fg="yellow")
                    sys.exit(0)
                if selected_model:  
                    break
                click.secho(
                    "No model selected. Please try again or press 'q' to exit.",
                    fg="red",
                )
            env_vars["MODEL"] = selected_model

        if selected_provider in ENV_VARS:
            provider_env_vars = ENV_VARS[selected_provider]
            for details in provider_env_vars:
                if details.get("default", False):
                    for key, value in details.items():
                        if key not in ["prompt", "key_name", "default"]:
                            env_vars[key] = value
                elif "key_name" in details:
                    prompt = details["prompt"]
                    key_name = details["key_name"]
                    api_key_value = click.prompt(prompt, default="", show_default=False)

                    if api_key_value.strip():
                        env_vars[key_name] = api_key_value

        if env_vars:
            write_env_file(folder_path, env_vars)
            click.secho("API keys and model saved to .env file", fg="green")
        else:
            click.secho(
                "No API keys provided or required by provider. Skipping .env file creation.", fg="yellow"
            )

        click.secho(f"Selected model: {env_vars.get('MODEL', 'N/A')}", fg="green")

    copy_template_files(folder_path, name, class_name, parent_folder)

    click.secho(f"Crew '{name}' created successfully!", fg="green")
    click.secho(f"To run your crew, cd into '{folder_name}' and run 'crewai run'", fg="cyan")
