from pathlib import Path
import click
from crewai.cli.utils import copy_template, load_env_vars, write_env_file
from crewai.cli.provider import get_provider_data, select_provider, PROVIDERS
from crewai.cli.constants import ENV_VARS


def create_folder_structure(name, parent_folder=None):
    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    if parent_folder:
        folder_path = Path(parent_folder) / folder_name
    else:
        folder_path = Path(folder_name)

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
    else:
        click.secho(
            f"\tFolder {folder_name} already exists.",
            fg="yellow",
        )

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


def create_crew(name, parent_folder=None):
    folder_path, folder_name, class_name = create_folder_structure(name, parent_folder)
    env_vars = load_env_vars(folder_path)

    provider_models = get_provider_data()
    if not provider_models:
        return

    selected_provider = select_provider(provider_models)
    if not selected_provider:
        return
    provider = selected_provider

    # selected_model = select_model(provider, provider_models)
    # if not selected_model:
    #     return
    # model = selected_model

    if provider in PROVIDERS:
        api_key_var = ENV_VARS[provider][0]
    else:
        api_key_var = click.prompt(
            f"Enter the environment variable name for your {provider.capitalize()} API key",
            type=str,
        )

    env_vars = {api_key_var: "YOUR_API_KEY_HERE"}
    write_env_file(folder_path, env_vars)

    # env_vars['MODEL'] = model
    # click.secho(f"Selected model: {model}", fg="green")

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
