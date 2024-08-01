import os
import shutil
from pathlib import Path

import click


def create_pipeline(name, router=False):
    """Create a new pipeline project."""
    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    click.secho(f"Creating pipeline {folder_name}...", fg="green", bold=True)

    project_root = Path(folder_name)
    if project_root.exists():
        click.secho(f"Error: Folder {folder_name} already exists.", fg="red")
        return

    package_dir = Path(__file__).parent
    template_folder = "pipeline_router" if router else "pipeline"
    templates_dir = package_dir / "templates" / template_folder

    # Copy the entire template directory structure
    shutil.copytree(templates_dir, project_root)

    # Process and replace placeholders in specific files
    files_to_process = [
        "README.md",
        "pyproject.toml",
        "main.py",
        "pipeline.py",
        "crews/research_crew.py",
        "crews/write_x_crew.py",
        "crews/write_linkedin_crew.py",
    ]

    for file_path in files_to_process:
        full_path = project_root / file_path
        if full_path.exists():
            with open(full_path, "r") as file:
                content = file.read()

            content = content.replace("{{name}}", name)
            content = content.replace("{{crew_name}}", class_name)
            content = content.replace("{{folder_name}}", folder_name)
            content = content.replace("{{pipeline_name}}", class_name)  # Add this line

            with open(full_path, "w") as file:
                file.write(content)

    click.secho(f"Pipeline {name} created successfully!", fg="green", bold=True)
