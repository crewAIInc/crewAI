from pathlib import Path
import subprocess
import zipfile

import pytest

from crewai_cli.deploy.archive import create_project_zip


def test_create_project_zip_excludes_local_artifacts(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "uv.lock").write_text("# lock\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / ".env").write_text("OPENAI_API_KEY=secret\n")
    (tmp_path / ".env.example").write_text("OPENAI_API_KEY=\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "main.pyc").write_bytes(b"compiled")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")

    archive_path = create_project_zip("demo", project_dir=tmp_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
    finally:
        archive_path.unlink(missing_ok=True)

    assert names == {
        "pyproject.toml",
        "uv.lock",
        "src/main.py",
        ".env.example",
    }


def test_create_project_zip_uses_repository_file_list(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "uv.lock").write_text("# lock\n")
    (tmp_path / "ignored.txt").write_text("ignored\n")

    class RepositoryStub:
        def deployable_files(self) -> list[str]:
            return ["pyproject.toml", "uv.lock"]

    archive_path = create_project_zip(
        "demo",
        project_dir=tmp_path,
        repository=RepositoryStub(),  # type: ignore[arg-type]
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
    finally:
        archive_path.unlink(missing_ok=True)

    assert names == {"pyproject.toml", "uv.lock"}


def test_create_project_zip_without_repository_uses_git_ignore_rules(
    tmp_path: Path,
):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / ".gitignore").write_text("node_modules/\nsecret.txt\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "package.json").write_text("{}\n")
    (tmp_path / "secret.txt").write_text("secret\n")

    try:
        subprocess.run(
            ["git", "init"],
            cwd=tmp_path,
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"git is not available in this environment: {exc}")

    archive_path = create_project_zip("demo", project_dir=tmp_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
    finally:
        archive_path.unlink(missing_ok=True)

    assert names == {
        ".gitignore",
        "pyproject.toml",
        "src/main.py",
    }


def test_create_project_zip_does_not_fallback_when_repository_listing_fails(
    tmp_path: Path,
):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")

    class RepositoryStub:
        def deployable_files(self) -> list[str]:
            raise RuntimeError("git listing failed")

    with pytest.raises(RuntimeError, match="git listing failed"):
        create_project_zip(
            "demo",
            project_dir=tmp_path,
            repository=RepositoryStub(),  # type: ignore[arg-type]
        )


def test_create_project_zip_excludes_symlinked_files(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    outside_file = tmp_path.parent / f"{tmp_path.name}-secret.txt"
    outside_file.write_text("secret\n")
    archive_path: Path | None = None
    try:
        try:
            (tmp_path / "external-secret.txt").symlink_to(outside_file)
        except OSError as exc:
            pytest.skip(f"symlinks are not supported in this environment: {exc}")

        archive_path = create_project_zip("demo", project_dir=tmp_path)
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        outside_file.unlink(missing_ok=True)

    assert names == {"pyproject.toml"}


def test_create_project_zip_adds_json_project_wrapper(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "json_crew"
version = "0.1.0"
dependencies = ["crewai[tools]>=1.15"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.crewai]
type = "crew"
""".strip()
        + "\n"
    )
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "researcher.jsonc").write_text("{}\n")
    (tmp_path / "crew.jsonc").write_text("{}\n")

    archive_path = create_project_zip("json_crew", project_dir=tmp_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            crew_py = archive.read("src/json_crew/crew.py").decode()
            main_py = archive.read("src/json_crew/main.py").decode()
            pyproject = archive.read("pyproject.toml").decode()
    finally:
        archive_path.unlink(missing_ok=True)

    assert "uv.lock" not in names
    assert "crew.jsonc" in names
    assert "agents/researcher.jsonc" in names
    assert "src/json_crew/__init__.py" in names
    assert "src/json_crew/crew.py" in names
    assert "src/json_crew/main.py" in names
    assert "src/json_crew/config/agents.yaml" in names
    assert "src/json_crew/config/tasks.yaml" in names
    assert "load_crew(_crew_path())" in crew_py
    assert "JsonCrew" in crew_py
    assert "from json_crew.crew import JsonCrew" in main_py
    assert "run_crew = \"json_crew.main:run\"" in pyproject


@pytest.mark.parametrize(
    "tool_config",
    [
        'tool = "invalid"\n',
        '[tool]\ncrewai = "invalid"\n',
    ],
)
def test_create_project_zip_adds_json_wrapper_for_malformed_tool_config(
    tmp_path: Path, tool_config: str
):
    (tmp_path / "pyproject.toml").write_text(
        f"""
[project]
name = "json_crew"
version = "0.1.0"

{tool_config}
""".strip()
        + "\n"
    )
    (tmp_path / "crew.jsonc").write_text("{}\n")

    archive_path = create_project_zip("json_crew", project_dir=tmp_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            pyproject = archive.read("pyproject.toml").decode()
    finally:
        archive_path.unlink(missing_ok=True)

    assert "src/json_crew/crew.py" in names
    assert "src/json_crew/main.py" in names
    assert "run_crew = \"json_crew.main:run\"" in pyproject


def test_create_project_zip_rejects_empty_normalized_package_name(tmp_path: Path):
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "!!!"
version = "0.1.0"

[tool.crewai]
type = "crew"
""".strip()
        + "\n"
    )
    (tmp_path / "crew.jsonc").write_text("{}\n")

    with pytest.raises(
        ValueError,
        match=r"Could not derive a valid Python package name",
    ):
        create_project_zip("invalid", project_dir=tmp_path)
