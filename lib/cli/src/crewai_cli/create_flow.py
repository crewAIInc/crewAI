from pathlib import Path
import shutil

import click
from crewai_core.telemetry import Telemetry

from crewai_cli.version import get_crewai_tools_dependency


DECLARATIVE_FLOW_FOLDERS = ("crews", "tools", "knowledge", "skills")


def create_flow(name: str, *, declarative: bool = False) -> None:
    """Create a new flow."""
    folder_name = name.replace(" ", "_").replace("-", "_").lower()
    class_name = name.replace("_", " ").replace("-", " ").title().replace(" ", "")

    click.secho(f"Creating flow {folder_name}...", fg="green", bold=True)

    project_root = Path(folder_name)
    if project_root.exists():
        click.secho(f"Error: Folder {folder_name} already exists.", fg="red")
        return

    telemetry = Telemetry()
    telemetry.flow_creation_span(class_name)

    if declarative:
        _create_declarative_flow(name, class_name, folder_name, project_root)
    else:
        _create_python_flow(name, class_name, folder_name, project_root)

    click.secho(f"Flow {name} created successfully!", fg="green", bold=True)


def _create_python_flow(
    name: str, class_name: str, folder_name: str, project_root: Path
) -> None:
    (project_root / "src" / folder_name).mkdir(parents=True)
    (project_root / "src" / folder_name / "crews").mkdir(parents=True)
    (project_root / "src" / folder_name / "tools").mkdir(parents=True)
    (project_root / "tests").mkdir(exist_ok=True)

    with open(project_root / ".env", "w") as file:
        file.write("OPENAI_API_KEY=YOUR_API_KEY")

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "flow"

    agents_md_src = package_dir / "templates" / "AGENTS.md"
    if agents_md_src.exists():
        shutil.copy2(agents_md_src, project_root / "AGENTS.md")

    root_template_files = [".gitignore", "pyproject.toml", "README.md"]
    src_template_files = ["__init__.py", "main.py"]
    tools_template_files = ["tools/__init__.py", "tools/custom_tool.py"]

    crew_folders = [
        "content_crew",
    ]

    def process_file(src_file: Path, dst_file: Path) -> None:
        if src_file.suffix in [".pyc", ".pyo", ".pyd"]:
            return

        try:
            with open(src_file, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as e:
            click.secho(f"Error processing file {src_file}: {e}", fg="red")
            return

        content = content.replace("{{name}}", name)
        content = content.replace("{{flow_name}}", class_name)
        content = content.replace("{{folder_name}}", folder_name)
        content = content.replace(
            "{{crewai_tools_dependency}}", get_crewai_tools_dependency()
        )

        with open(dst_file, "w") as file:
            file.write(content)

    for file_name in root_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / file_name
        process_file(src_file, dst_file)

    for file_name in src_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    for file_name in tools_template_files:
        src_file = templates_dir / file_name
        dst_file = project_root / "src" / folder_name / file_name
        process_file(src_file, dst_file)

    for crew_folder in crew_folders:
        src_crew_folder = templates_dir / "crews" / crew_folder
        dst_crew_folder = project_root / "src" / folder_name / "crews" / crew_folder
        if src_crew_folder.exists():
            for src_file in src_crew_folder.rglob("*"):
                if src_file.is_file():
                    relative_path = src_file.relative_to(src_crew_folder)
                    dst_file = dst_crew_folder / relative_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    process_file(src_file, dst_file)
        else:
            click.secho(
                f"Warning: Crew folder {crew_folder} not found in template.",
                fg="yellow",
            )


def _create_declarative_flow(
    name: str, class_name: str, folder_name: str, project_root: Path
) -> None:
    project_root.mkdir(parents=True)
    package_root = project_root / "src" / folder_name
    package_root.mkdir(parents=True)
    for folder in DECLARATIVE_FLOW_FOLDERS:
        (package_root / folder).mkdir()

    package_dir = Path(__file__).parent
    templates_dir = package_dir / "templates" / "declarative_flow"

    agents_md_src = package_dir / "templates" / "AGENTS.md"
    if agents_md_src.exists():
        shutil.copy2(agents_md_src, project_root / "AGENTS.md")

    for src_file in templates_dir.rglob("*"):
        if not src_file.is_file():
            continue

        relative_path = src_file.relative_to(templates_dir)
        dst_file = (
            project_root / relative_path
            if relative_path.name in {".gitignore", "README.md", "pyproject.toml"}
            else package_root / relative_path
        )
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        content = src_file.read_text(encoding="utf-8")
        content = content.replace("{{name}}", name)
        content = content.replace("{{flow_name}}", class_name)
        content = content.replace("{{folder_name}}", folder_name)
        content = content.replace(
            "{{crewai_tools_dependency}}", get_crewai_tools_dependency()
        )
        dst_file.write_text(content, encoding="utf-8")

    (project_root / ".env").write_text("OPENAI_API_KEY=YOUR_API_KEY", encoding="utf-8")
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    for folder in DECLARATIVE_FLOW_FOLDERS:
        (package_root / folder / ".gitkeep").write_text("", encoding="utf-8")
