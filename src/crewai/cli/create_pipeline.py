import os
from pathlib import Path

import click

from crewai.cli.create_crew import create_crew
from crewai.cli.utils import copy_template


def create_pipeline(pipeline_name, crew_names):
    """Create a new pipeline with multiple crews."""
    folder_name = pipeline_name.replace(" ", "_").replace("-", "_").lower()
    class_name = (
        pipeline_name.replace("_", " ").replace("-", " ").title().replace(" ", "")
    )

    click.secho(f"Creating pipeline {folder_name}...", fg="green", bold=True)

    # Create the main project structure
    project_root = Path(folder_name)
    project_root.mkdir(exist_ok=True)
    (project_root / "src" / folder_name).mkdir(parents=True, exist_ok=True)
    (project_root / "src" / folder_name / "crews").mkdir(parents=True, exist_ok=True)
    (project_root / "src" / folder_name / "config").mkdir(parents=True, exist_ok=True)
    (project_root / "src" / folder_name / "tools").mkdir(parents=True, exist_ok=True)
    (project_root / "tests").mkdir(exist_ok=True)

    # Create .env file
    with open(project_root / ".env", "w") as file:
        file.write("OPENAI_API_KEY=YOUR_API_KEY")

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "pipeline"

    # Process main.py template
    with open(templates_dir / "main.py", "r") as f:
        main_content = f.read()

    # Replace variables in main.py
    main_content = main_content.replace("{{folder_name}}", folder_name)
    main_content = main_content.replace("{{pipeline_name}}", class_name)

    # Write updated main.py
    with open(project_root / "src" / folder_name / "main.py", "w") as f:
        f.write(main_content)

    # Process pipeline.py template
    with open(templates_dir / "pipeline.py", "r") as f:
        pipeline_content = f.read()

    # Replace pipeline name
    pipeline_content = pipeline_content.replace("{{pipeline_name}}", class_name)

    # Generate crew initialization lines
    crew_init_lines = []
    crew_stage_lines = []
    for crew_name in crew_names:
        crew_class_name = (
            crew_name.replace("_", " ").replace("-", " ").title().replace(" ", "")
        )
        crew_init_lines.append(
            f"        self.{crew_name.lower()}_crew = {crew_class_name}Crew().crew()"
        )
        crew_stage_lines.append(f"                self.{crew_name.lower()}_crew,")

    # Replace crew initialization placeholder
    pipeline_content = pipeline_content.replace(
        "        {% for crew_name in crew_names %}\n"
        "        self.{{crew_name.lower()}}_crew = {{crew_name}}Crew().crew()\n"
        "        {% endfor %}",
        "\n".join(crew_init_lines),
    )

    # Replace crew stages placeholder
    pipeline_content = pipeline_content.replace(
        "                {% for crew_name in crew_names %}\n"
        "                self.{{crew_name.lower()}}_crew,\n"
        "                {% endfor %}",
        "\n".join(crew_stage_lines),
    )

    # Update imports with correct package structure
    crew_imports = [
        f"from {folder_name}.src.{folder_name}.crews.{name.lower()}.crew import {name.replace('_', ' ').replace('-', ' ').title().replace(' ', '')}Crew"
        for name in crew_names
    ]
    pipeline_content = pipeline_content.replace(
        "from crews.crew import *", "\n".join(crew_imports)
    )

    with open(project_root / "src" / folder_name / "pipeline.py", "w") as f:
        f.write(pipeline_content)

    # Copy and process other template files
    template_files = [
        (".gitignore", project_root),
        ("pyproject.toml", project_root),
        ("README.md", project_root),
        ("__init__.py", project_root / "src" / folder_name),
        ("tools/custom_tool.py", project_root / "src" / folder_name / "tools"),
        ("tools/__init__.py", project_root / "src" / folder_name / "tools"),
        ("config/agents.yaml", project_root / "src" / folder_name / "config"),
        ("config/tasks.yaml", project_root / "src" / folder_name / "config"),
    ]

    for template_file, destination in template_files:
        src_file = templates_dir / template_file
        dst_file = destination / os.path.basename(template_file)
        copy_template(src_file, dst_file, pipeline_name, class_name, folder_name)

    # Create crew files
    for crew_name in crew_names:
        create_crew(crew_name, project_root / "src" / folder_name / "crews")

    click.secho(
        f"Pipeline {pipeline_name} created successfully with crews: {', '.join(crew_names)}!",
        fg="green",
        bold=True,
    )
