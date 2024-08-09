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
    (project_root / "src" / folder_name / "pipelines").mkdir(parents=True)
    (project_root / "src" / folder_name / "crews").mkdir(parents=True)
    (project_root / "src" / folder_name / "tools").mkdir(parents=True)
    (project_root / "tests").mkdir(exist_ok=True)

    # Create .env file
    with open(project_root / ".env", "w") as file:
        file.write("OPENAI_API_KEY=YOUR_API_KEY")

    package_dir = Path(__file__).parent
    template_folder = "pipeline_router" if router else "pipeline"
    templates_dir = package_dir / "templates" / template_folder

    # List of template files to copy
    root_template_files = [".gitignore", "pyproject.toml", "README.md"]
    src_template_files = ["__init__.py", "main.py"]
    tools_template_files = ["tools/__init__.py", "tools/custom_tool.py"]

    if router:
        crew_folders = [
            "classifier_crew",
            "normal_crew",
            "urgent_crew",
        ]
        pipelines_folders = [
            "pipelines/__init__.py",
            "pipelines/pipeline_classifier.py",
            "pipelines/pipeline_normal.py",
            "pipelines/pipeline_urgent.py",
        ]
    else:
        crew_folders = [
            "research_crew",
            "write_linkedin_crew",
            "write_x_crew",
        ]
        pipelines_folders = ["pipelines/__init__.py", "pipelines/pipeline.py"]

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

    # Copy tools files
    for file_name in tools_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        shutil.copy(src_file, dst_file)

    # Copy pipelines folders
    for file_name in pipelines_folders:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    # Copy crew folders
    for crew_folder in crew_folders:
        src_crew_folder = templates_dir / "crews" / crew_folder
        dst_crew_folder = project_root / "src" / folder_name / "crews" / crew_folder
        if src_crew_folder.exists():
            shutil.copytree(src_crew_folder, dst_crew_folder)
        else:
            click.secho(
                f"Warning: Crew folder {crew_folder} not found in template.",
                fg="yellow",
            )

    click.secho(f"Pipeline {name} created successfully!", fg="green", bold=True)
