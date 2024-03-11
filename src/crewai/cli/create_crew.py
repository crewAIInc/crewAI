import os
from pathlib import Path

import click


def create_crew(name):
    """Create a new crew."""
    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    click.secho(f"Creating folder {folder_name}...", fg="green", bold=True)

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
        os.mkdir(folder_name + "/tests")
        os.mkdir(folder_name + "/src")
        os.mkdir(folder_name + f"/src/{folder_name}")
        os.mkdir(folder_name + f"/src/{folder_name}/tools")
        os.mkdir(folder_name + f"/src/{folder_name}/config")
        with open(folder_name + "/.env", "w") as file:
            file.write("OPENAI_API_KEY=YOUR_API_KEY")
    else:
        click.secho(
            f"\tFolder {folder_name} already exists. Please choose a different name.",
            fg="red",
        )
        return

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates"

    # List of template files to copy
    root_template_files = [
        ".gitignore",
        "pyproject.toml",
        "README.md",
    ]
    tools_template_files = ["tools/custom_tool.py", "tools/__init__.py"]
    config_template_files = ["config/agents.yaml", "config/tasks.yaml"]
    src_template_files = ["__init__.py", "main.py", "crew.py"]

    for file_name in root_template_files:
        src_file = templates_dir / file_name
        dst_file = Path(folder_name) / file_name
        copy_template(src_file, dst_file, name, class_name, folder_name)

    for file_name in src_template_files:
        src_file = templates_dir / file_name
        dst_file = Path(folder_name) / "src" / folder_name / file_name
        copy_template(src_file, dst_file, name, class_name, folder_name)

    for file_name in tools_template_files:
        src_file = templates_dir / file_name
        dst_file = Path(folder_name) / "src" / folder_name / file_name
        copy_template(src_file, dst_file, name, class_name, folder_name)

    for file_name in config_template_files:
        src_file = templates_dir / file_name
        dst_file = Path(folder_name) / "src" / folder_name / file_name
        copy_template(src_file, dst_file, name, class_name, folder_name)

    click.secho(f"Crew {name} created successfully!", fg="green", bold=True)


def copy_template(src, dst, name, class_name, folder_name):
    """Copy a file from src to dst."""
    with open(src, "r") as file:
        content = file.read()

    # Interpolate the content
    content = content.replace("{{name}}", name)
    content = content.replace("{{crew_name}}", class_name)
    content = content.replace("{{folder_name}}", folder_name)

    # Write the interpolated content to the new file
    with open(dst, "w") as file:
        file.write(content)

    click.secho(f"  - Created {dst}", fg="green")
