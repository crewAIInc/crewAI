from pathlib import Path

import click

from crewai.cli.utils import copy_template


def create_crew(name, parent_folder=None):
    """Create a new crew."""
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

    # Create necessary directories
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        (folder_path / "tests").mkdir(exist_ok=True)
        if not parent_folder:
            (folder_path / "src" / folder_name).mkdir(parents=True)
            (folder_path / "src" / folder_name / "tools").mkdir(parents=True)
            (folder_path / "src" / folder_name / "config").mkdir(parents=True)
    else:
        click.secho(
            f"\tFolder {folder_name} already exists. Updating .env file...",
            fg="yellow",
        )

    # Path to the .env file
    env_file_path = folder_path / ".env"

    # Load existing environment variables if .env exists
    env_vars = {}
    if env_file_path.exists():
        with open(env_file_path, "r") as file:
            for line in file:
                key_value = line.strip().split('=', 1)
                if len(key_value) == 2:
                    env_vars[key_value[0]] = key_value[1]

    # Prompt for keys/variables/LLM settings only if not already set
    if 'OPENAI_API_KEY' not in env_vars:
        if click.confirm("Do you want to enter your OPENAI_API_KEY?", default=True):
            env_vars['OPENAI_API_KEY'] = click.prompt("Enter your OPENAI_API_KEY", type=str)

    if 'ANTHROPIC_API_KEY' not in env_vars:
        if click.confirm("Do you want to enter your ANTHROPIC_API_KEY?", default=False):
            env_vars['ANTHROPIC_API_KEY'] = click.prompt("Enter your ANTHROPIC_API_KEY", type=str)

    if 'GEMINI_API_KEY' not in env_vars:
        if click.confirm("Do you want to specify your GEMINI_API_KEY?", default=True):
            env_vars['GEMINI_API_KEY'] = click.prompt("Enter your GEMINI_API_KEY", type=str)

    # Loop to add other environment variables
    while click.confirm("Do you want to specify another environment variable?", default=False):
        var_name = click.prompt("Enter the variable name", type=str)
        var_value = click.prompt(f"Enter the value for {var_name}", type=str)
        env_vars[var_name] = var_value

    # Write the environment variables to .env file
    with open(env_file_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "crew"

    # List of template files to copy
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
