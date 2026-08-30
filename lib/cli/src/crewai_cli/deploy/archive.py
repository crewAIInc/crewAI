from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import zipfile

from crewai_cli import git


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
    """Return project-relative files to include in the archive."""
    if repository is not None:
        return _repository_project_files(root, repository)

    try:
        repository = git.Repository(path=str(root), fetch=False)
    except ValueError:
        repository = None

    if repository is not None:
        return _repository_project_files(root, repository)

    return [
        path
        for path in _walk_files(root)
        if not _is_excluded(path) and _is_regular_file(root / path)
    ]


def _repository_project_files(root: Path, repository: git.Repository) -> list[Path]:
    """Return deployable files from Git while applying local safety excludes."""
    files = [Path(path) for path in repository.deployable_files()]
    return [
        path
        for path in files
        if not _is_excluded(path) and _is_regular_file(root / path)
    ]


def _walk_files(root: Path) -> list[Path]:
    """List regular files below root as project-relative paths."""
    return [
        path.relative_to(root) for path in root.rglob("*") if _is_regular_file(path)
    ]


def _is_regular_file(path: Path) -> bool:
    """Return True for regular files, excluding symlinks to files."""
    return path.is_file() and not path.is_symlink()


def _is_excluded(path: Path) -> bool:
    """Return True when a file should be omitted from deployment ZIPs."""
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
    """Copy archive files into a temporary staging directory."""
    staging_root = Path(tempfile.mkdtemp(prefix="crewai-deploy-"))

    try:
        for relative_path in files:
            source = root / relative_path
            if not _is_regular_file(source):
                continue

            destination = staging_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return staging_root
