from __future__ import annotations

from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import zipfile

from crewai_cli import git
from crewai_cli.utils import parse_toml


_EXCLUDED_DIRS = {
    ".crewai",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "venv",
}
_EXCLUDED_FILES = {
    ".DS_Store",
    ".env",
}
_ALLOWED_ENV_EXAMPLES = {
    ".env.example",
    ".env.sample",
}
_EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def create_project_zip(
    project_name: str,
    *,
    project_dir: Path | None = None,
    repository: git.Repository | None = None,
) -> Path:
    """Create a deployable ZIP archive for a CrewAI project."""
    root = (project_dir or Path.cwd()).resolve()
    files = _project_files(root, repository)
    if not files:
        raise ValueError("No deployable project files were found.")

    staged_root = _stage_project(root, files)
    archive_handle = tempfile.NamedTemporaryFile(
        prefix=f"{project_name}-",
        suffix=".zip",
        delete=False,
    )
    archive_path = Path(archive_handle.name)
    archive_handle.close()

    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for relative_path in _walk_files(staged_root):
                absolute_path = staged_root / relative_path
                zip_file.write(absolute_path, relative_path.as_posix())
    finally:
        shutil.rmtree(staged_root, ignore_errors=True)

    return archive_path


def _project_files(root: Path, repository: git.Repository | None = None) -> list[Path]:
    if repository is not None:
        try:
            files = [Path(path) for path in repository.deployable_files()]
            return [
                path
                for path in files
                if not _is_excluded(path) and (root / path).is_file()
            ]
        except Exception:  # noqa: S110
            pass

    return [
        path
        for path in _walk_files(root)
        if not _is_excluded(path) and (root / path).is_file()
    ]


def _walk_files(root: Path) -> list[Path]:
    return [path.relative_to(root) for path in root.rglob("*") if path.is_file()]


def _is_excluded(path: Path) -> bool:
    parts = set(path.parts)
    if parts.intersection(_EXCLUDED_DIRS):
        return True

    name = path.name
    if name in _EXCLUDED_FILES:
        return True
    if name.startswith(".env.") and name not in _ALLOWED_ENV_EXAMPLES:
        return True
    return path.suffix in _EXCLUDED_SUFFIXES


def _stage_project(root: Path, files: list[Path]) -> Path:
    staging_root = Path(tempfile.mkdtemp(prefix="crewai-deploy-"))

    try:
        for relative_path in files:
            destination = staging_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / relative_path, destination)

        if _is_json_crew_project(staging_root):
            _add_json_crew_deploy_wrapper(staging_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staging_root


def _is_json_crew_project(root: Path) -> bool:
    if not ((root / "crew.jsonc").is_file() or (root / "crew.json").is_file()):
        return False

    project = _read_pyproject(root)
    tool_config = project.get("tool") or {}
    if not isinstance(tool_config, dict):
        return False

    crewai_config = tool_config.get("crewai") or {}
    if not isinstance(crewai_config, dict):
        return False

    declared_type = crewai_config.get("type")
    if declared_type == "flow":
        return False

    package_name = _package_name(root)
    if package_name is None:
        return False

    return not (root / "src" / package_name / "crew.py").is_file()


def _read_pyproject(root: Path) -> dict[str, Any]:
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        return {}
    try:
        return parse_toml(pyproject_path.read_text())
    except Exception:
        return {}


def _package_name(root: Path) -> str | None:
    project = _read_pyproject(root).get("project")
    if not isinstance(project, dict):
        return None

    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    folder = name.replace(" ", "_").replace("-", "_").lower()
    return re.sub(r"[^a-zA-Z0-9_]", "", folder)


def _class_name(package_name: str) -> str:
    parts = [part for part in re.split(r"[^a-zA-Z0-9]+", package_name) if part]
    class_name = "".join(part[:1].upper() + part[1:] for part in parts)
    if not class_name:
        return "JsonCrew"
    if class_name[0].isdigit():
        return f"Crew{class_name}"
    return class_name


def _add_json_crew_deploy_wrapper(root: Path) -> None:
    package_name = _package_name(root)
    if package_name is None:
        return

    package_dir = root / "src" / package_name
    config_dir = package_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    class_name = _class_name(package_name)
    crew_filename = "crew.jsonc" if (root / "crew.jsonc").is_file() else "crew.json"

    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (config_dir / "agents.yaml").write_text("{}\n", encoding="utf-8")
    (config_dir / "tasks.yaml").write_text("{}\n", encoding="utf-8")
    (package_dir / "crew.py").write_text(
        _json_crew_py(class_name, crew_filename),
        encoding="utf-8",
    )
    (package_dir / "main.py").write_text(
        _json_main_py(package_name, class_name),
        encoding="utf-8",
    )
    _ensure_project_scripts(root, package_name)


def _json_crew_py(class_name: str, crew_filename: str) -> str:
    return f'''from pathlib import Path

from crewai import Crew
from crewai.project import CrewBase, crew
from crewai.project.crew_loader import load_crew


def _crew_path() -> Path:
    return Path(__file__).resolve().parents[2] / "{crew_filename}"


@CrewBase
class {class_name}:
    """Compatibility wrapper for a JSON-defined CrewAI project."""

    @crew
    def crew(self) -> Crew:
        crew_instance, default_inputs = load_crew(_crew_path())
        self.default_inputs = default_inputs
        return crew_instance
'''


def _json_main_py(package_name: str, class_name: str) -> str:
    return f"""#!/usr/bin/env python
import json
import sys

from {package_name}.crew import {class_name}


def _load():
    wrapper = {class_name}()
    crew = wrapper.crew()
    return crew, getattr(wrapper, "default_inputs", {{}})


def run():
    crew, inputs = _load()
    return crew.kickoff(inputs=inputs)


def train():
    crew, inputs = _load()
    return crew.train(
        n_iterations=int(sys.argv[1]),
        filename=sys.argv[2],
        inputs=inputs,
    )


def replay():
    crew, _ = _load()
    return crew.replay(task_id=sys.argv[1])


def test():
    crew, inputs = _load()
    return crew.test(
        n_iterations=int(sys.argv[1]),
        eval_llm=sys.argv[2],
        inputs=inputs,
    )


def run_with_trigger():
    if len(sys.argv) < 2:
        raise ValueError("No trigger payload provided.")

    crew, inputs = _load()
    trigger_payload = json.loads(sys.argv[1])
    return crew.kickoff(
        inputs={{**inputs, "crewai_trigger_payload": trigger_payload}}
    )
"""


def _ensure_project_scripts(root: Path, package_name: str) -> None:
    pyproject_path = root / "pyproject.toml"
    if not pyproject_path.is_file():
        return

    content = pyproject_path.read_text(encoding="utf-8")
    if "[project.scripts]" in content:
        return

    script_block = f'''

[project.scripts]
{package_name} = "{package_name}.main:run"
run_crew = "{package_name}.main:run"
train = "{package_name}.main:train"
replay = "{package_name}.main:replay"
test = "{package_name}.main:test"
run_with_trigger = "{package_name}.main:run_with_trigger"
'''
    pyproject_path.write_text(content.rstrip() + script_block, encoding="utf-8")
