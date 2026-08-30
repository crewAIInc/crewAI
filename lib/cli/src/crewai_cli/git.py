from __future__ import annotations

from functools import cached_property
from pathlib import Path
import subprocess


_INITIAL_COMMIT_EXCLUDE_PATTERNS = [
    ".crewai/",
    ".env",
    ".env.*",
    "!.env.example",
    "!.env.sample",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    ".venv/",
    "__pycache__/",
    "build/",
    "dist/",
    "env/",
    "venv/",
]


class Repository:
    def __init__(self, path: str = ".", fetch: bool = True) -> None:
        self.path = path

        if not self.is_git_installed():
            raise ValueError("Git is not installed or not found in your PATH.")

        if not self.is_git_repo:
            raise ValueError(f"{self.path} is not a Git repository.")

        if fetch:
            self.fetch()

    @staticmethod
    def is_git_installed() -> bool:
        """Check if Git is installed and available in the system."""
        try:
            subprocess.run(
                ["git", "--version"],  # noqa: S607
                capture_output=True,
                check=True,
                text=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def fetch(self) -> None:
        """Fetch latest updates from the remote."""
        command = ["git", "fetch"]
        result = subprocess.run(  # noqa: S603
            command,
            cwd=self.path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        if "No remote repository specified" in result.stderr:
            return
        details = result.stderr.strip() or result.stdout.strip() or "no output"
        raise ValueError(
            f"Git fetch failed with exit code {result.returncode} "
            f"for command {command!r}: {details}"
        )

    @classmethod
    def initialize(cls, path: str = ".") -> Repository:
        """Initialize a Git repository and create an initial commit if needed."""
        if not cls.is_git_installed():
            raise ValueError("Git is not installed or not found in your PATH.")

        subprocess.run(["git", "init"], cwd=path, check=True)  # noqa: S607
        repository = cls(path=path, fetch=False)
        repository.create_initial_commit_if_needed()
        return repository

    def status(self) -> str:
        """Get the git status in porcelain format."""
        return subprocess.check_output(
            ["git", "status", "--branch", "--porcelain"],  # noqa: S607
            cwd=self.path,
            encoding="utf-8",
        ).strip()

    @cached_property
    def is_git_repo(self) -> bool:
        """Check if the current directory is a git repository."""
        try:
            subprocess.check_output(
                ["git", "rev-parse", "--is-inside-work-tree"],  # noqa: S607
                cwd=self.path,
                encoding="utf-8",
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def has_uncommitted_changes(self) -> bool:
        """Check if the repository has uncommitted changes."""
        return len(self.status().splitlines()) > 1

    def is_ahead_or_behind(self) -> bool:
        """Check if the repository is ahead or behind the remote."""
        for line in self.status().splitlines():
            if line.startswith("##") and ("ahead" in line or "behind" in line):
                return True
        return False

    def is_synced(self) -> bool:
        """Return True if the Git repository is fully synced with the remote, False otherwise."""
        if self.has_uncommitted_changes() or self.is_ahead_or_behind():
            return False
        return True

    def has_commits(self) -> bool:
        """Return True if the repository has at least one commit."""
        try:
            subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"],  # noqa: S607
                cwd=self.path,
                capture_output=True,
                check=True,
                text=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def create_initial_commit_if_needed(self) -> bool:
        """Create a local initial commit when the repository has no commits."""
        if self.has_commits():
            return False

        self._ensure_initial_commit_excludes()
        subprocess.run(["git", "add", "."], cwd=self.path, check=True)  # noqa: S607
        command = [
            "git",
            "-c",
            "user.name=CrewAI",
            "-c",
            "user.email=deploy@crewai.com",
            "commit",
            "--allow-empty",
            "-m",
            "Initial crew",
        ]
        subprocess.run(  # noqa: S603
            command,
            cwd=self.path,
            check=True,
        )
        return True

    def _ensure_initial_commit_excludes(self) -> None:
        """Add local-only ignore patterns before auto-staging an initial commit."""
        exclude_file = Path(self.path) / ".git" / "info" / "exclude"
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        existing = exclude_file.read_text() if exclude_file.exists() else ""
        existing_lines = set(existing.splitlines())
        missing_patterns = [
            pattern
            for pattern in _INITIAL_COMMIT_EXCLUDE_PATTERNS
            if pattern not in existing_lines
        ]
        if not missing_patterns:
            return

        prefix = "" if existing.endswith("\n") or not existing else "\n"
        patterns = "\n".join(missing_patterns)
        exclude_file.write_text(
            f"{existing}{prefix}# CrewAI deploy auto-commit excludes\n{patterns}\n"
        )

    def deployable_files(self) -> list[str]:
        """Return files tracked by Git or untracked and not ignored."""
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],  # noqa: S607
            cwd=self.path,
            encoding="utf-8",
        )
        return [line for line in output.splitlines() if line]

    def origin_url(self) -> str | None:
        """Get the Git repository's remote URL."""
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],  # noqa: S607
                cwd=self.path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None


def initialize_if_git_available(path: Path) -> bool:
    """Initialize a Git repository when Git is available."""
    if not Repository.is_git_installed():
        return False

    subprocess.run(["git", "init"], cwd=path, check=True)  # noqa: S607
    return True
