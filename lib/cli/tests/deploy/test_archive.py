from pathlib import Path
import zipfile

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
