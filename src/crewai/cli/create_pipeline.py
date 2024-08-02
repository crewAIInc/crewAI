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

    # Create directory structure
    (project_root / "src" / folder_name).mkdir(parents=True)
    (project_root / "src" / folder_name / "crews").mkdir(parents=True)
    (project_root / "src" / folder_name / "tools").mkdir(parents=True)
    (project_root / "src" / folder_name / "config").mkdir(parents=True)
    (project_root / "tests").mkdir(exist_ok=True)

    # Create .env file
    with open(project_root / ".env", "w") as file:
        file.write("OPENAI_API_KEY=YOUR_API_KEY")

    package_dir = Path(__file__).parent
    template_folder = "pipeline_router" if router else "pipeline"
    templates_dir = package_dir / "templates" / template_folder

    # List of template files to copy
    root_template_files = [".gitignore", "pyproject.toml", "README.md"]
    src_template_files = ["__init__.py", "main.py", "pipeline.py"]
    tools_template_files = ["tools/custom_tool.py", "tools/__init__.py"]
    config_template_files = ["config/agents.yaml", "config/tasks.yaml"]
    crew_template_files = ["crews/research_crew.py", "crews/write_x_crew.py"]

    if router:
        crew_template_files.append("crews/write_linkedin_crew.py")

    def process_file(src_file, dst_file):
        with open(src_file, "r") as file:
            content = file.read()

        content = content.replace("{{name}}", name)
        content = content.replace("{{crew_name}}", class_name)
        content = content.replace("{{folder_name}}", folder_name)
        content = content.replace("{{pipeline_name}}", class_name)

        with open(dst_file, "w") as file:
            file.write(content)

    # Copy and process root template files
    for file_name in root_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / file_name
        process_file(src_file, dst_file)

    # Copy and process src template files
    for file_name in src_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    # Copy tools and config files
    for file_name in tools_template_files + config_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        shutil.copy(src_file, dst_file)

    # Copy and process crew files
    for file_name in crew_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    click.secho(f"Pipeline {name} created successfully!", fg="green", bold=True)
