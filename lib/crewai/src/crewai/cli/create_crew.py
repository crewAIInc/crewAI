from pathlib import Path
import shutil
import sys

import click

from crewai.cli.constants import ENV_VARS, MODELS
from crewai.cli.provider import (
    get_provider_data,
    select_model,
    select_provider,
)
from crewai.cli.utils import copy_template, load_env_vars, write_env_file


def create_folder_structure(name, parent_folder=None):
    import keyword
    import re

    name = name.rstrip("/")

    if not name.strip():
        raise ValueError("Projectnaam kan niet leeg zijn of alleen spaties bevatten")

    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    folder_name = re.sub(r"[^a-zA-Z0-9_]", "", folder_name)

    # Check if the name starts with invalid characters or is primarily invalid
    if re.match(r"^[^a-zA-Z0-9_-]+", name):
        raise ValueError(
            f"Projectnaam '{name}' bevat geen geldige karakters voor een Python module naam"
        )

    if not folder_name:
        raise ValueError(
            f"Projectnaam '{name}' bevat geen geldige karakters voor een Python module naam"
        )

    if folder_name[0].isdigit():
        raise ValueError(
            f"Projectnaam '{name}' zou mapnaam '{folder_name}' genereren die niet met een cijfer kan beginnen (ongeldige Python module naam)"
        )

    if keyword.iskeyword(folder_name):
        raise ValueError(
            f"Projectnaam '{name}' zou mapnaam '{folder_name}' genereren wat een gereserveerd Python sleutelwoord is"
        )

    if not folder_name.isidentifier():
        raise ValueError(
            f"Projectnaam '{name}' zou ongeldige Python module naam '{folder_name}' genereren"
        )

    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    class_name = re.sub(r"[^a-zA-Z0-9_]", "", class_name)

    if not class_name:
        raise ValueError(
            f"Projectnaam '{name}' bevat geen geldige karakters voor een Python klasse naam"
        )

    if class_name[0].isdigit():
        raise ValueError(
            f"Projectnaam '{name}' zou klassenaam '{class_name}' genereren die niet met een cijfer kan beginnen"
        )

    # Check if the original name (before title casing) is a keyword
    original_name_clean = re.sub(
        r"[^a-zA-Z0-9_]", "", name.replace("_", "").replace("-", "").lower()
    )
    if (
        keyword.iskeyword(original_name_clean)
        or keyword.iskeyword(class_name)
        or class_name in ("True", "False", "None")
    ):
        raise ValueError(
            f"Projectnaam '{name}' zou klassenaam '{class_name}' genereren wat een gereserveerd Python sleutelwoord is"
        )

    if not class_name.isidentifier():
        raise ValueError(
            f"Projectnaam '{name}' zou ongeldige Python klasse naam '{class_name}' genereren"
        )

    if parent_folder:
        folder_path = Path(parent_folder) / folder_name
    else:
        folder_path = Path(folder_name)

    if folder_path.exists():
        if not click.confirm(
            f"Map {folder_name} bestaat al. Wil je deze overschrijven?"
        ):
            click.secho("Operatie geannuleerd.", fg="yellow")
            sys.exit(0)
        click.secho(f"Map {folder_name} wordt overschreven...", fg="green", bold=True)
        shutil.rmtree(folder_path)  # Delete the existing folder and its contents

    click.secho(
        f"{'Crew' if parent_folder else 'Map'} {folder_name} wordt aangemaakt...",
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
        if not provider:
            provider_models = get_provider_data()
            if not provider_models:
                return

        existing_provider = None
        for provider, env_keys in ENV_VARS.items():
            if any(
                "key_name" in details and details["key_name"] in env_vars
                for details in env_keys
            ):
                existing_provider = provider
                break

        if existing_provider:
            if not click.confirm(
                f"Bestaande omgevingsvariabele configuratie gevonden voor {existing_provider.capitalize()}. Wil je deze overschrijven?"
            ):
                click.secho("Bestaande provider configuratie wordt behouden.", fg="yellow")
                return

        provider_models = get_provider_data()
        if not provider_models:
            return

        while True:
            selected_provider = select_provider(provider_models)
            if selected_provider is None:  # User typed 'q'
                click.secho("Afsluiten...", fg="yellow")
                sys.exit(0)
            if selected_provider:  # Valid selection
                break
            click.secho(
                "Geen provider geselecteerd. Probeer opnieuw of druk op 'q' om af te sluiten.", fg="red"
            )

        # Check if the selected provider has predefined models
        if MODELS.get(selected_provider):
            while True:
                selected_model = select_model(selected_provider, provider_models)
                if selected_model is None:  # User typed 'q'
                    click.secho("Afsluiten...", fg="yellow")
                    sys.exit(0)
                if selected_model:  # Valid selection
                    break
                click.secho(
                    "Geen model geselecteerd. Probeer opnieuw of druk op 'q' om af te sluiten.",
                    fg="red",
                )
            env_vars["MODEL"] = selected_model

        # Check if the selected provider requires API keys
        if selected_provider in ENV_VARS:
            provider_env_vars = ENV_VARS[selected_provider]
            for details in provider_env_vars:
                if details.get("default", False):
                    # Automatically add default key-value pairs
                    for key, value in details.items():
                        if key not in ["prompt", "key_name", "default"]:
                            env_vars[key] = value
                elif "key_name" in details:
                    # Prompt for non-default key-value pairs
                    prompt = details["prompt"]
                    key_name = details["key_name"]
                    api_key_value = click.prompt(prompt, default="", show_default=False)

                    if api_key_value.strip():
                        env_vars[key_name] = api_key_value

        if env_vars:
            write_env_file(folder_path, env_vars)
            click.secho("API sleutels en model opgeslagen in .env bestand", fg="green")
        else:
            click.secho(
                "Geen API sleutels opgegeven. .env bestand aanmaken overgeslagen.", fg="yellow"
            )

        click.secho(f"Geselecteerd model: {env_vars.get('MODEL', 'N.v.t.')}", fg="green")

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "crew"

    root_template_files = (
        [".gitignore", "pyproject.toml", "README.md", "knowledge/user_preference.txt"]
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

    click.secho(f"Crew {name} succesvol aangemaakt!", fg="green", bold=True)
