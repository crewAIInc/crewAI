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

    if not folder_path.exists():
        folder_path.mkdir(parents=True)
        (folder_path / "tests").mkdir(exist_ok=True)
        if not parent_folder:
            (folder_path / "src" / folder_name).mkdir(parents=True)
            (folder_path / "src" / folder_name / "tools").mkdir(parents=True)
            (folder_path / "src" / folder_name / "config").mkdir(parents=True)
            with open(folder_path / ".env", "w") as file:
                file.write("OPENAI_API_KEY=YOUR_API_KEY")
    else:
        click.secho(
            f"\tFolder {folder_name} already exists. Please choose a different name.",
            fg="red",
        )
        return

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
