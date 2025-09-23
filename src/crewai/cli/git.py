import subprocess
from functools import lru_cache


class Repository:
    def __init__(self, path="."):
        self.path = path

        if not self.is_git_installed():
            raise ValueError("Git is not installed or not found in your PATH.")

        if not self.is_git_repo():
            raise ValueError(f"{self.path} is not a Git repository.")

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
        subprocess.run(["git", "fetch"], cwd=self.path, check=True)  # noqa: S607

    def status(self) -> str:
        """Get the git status in porcelain format."""
        return subprocess.check_output(
            ["git", "status", "--branch", "--porcelain"],  # noqa: S607
            cwd=self.path,
            encoding="utf-8",
        ).strip()

    @lru_cache(maxsize=None)  # noqa: B019
    def is_git_repo(self) -> bool:
        """Check if the current directory is a git repository.

        Notes:
          - TODO: This method is cached to avoid redundant checks, but using lru_cache on methods can lead to memory leaks
        """
        try:
            subprocess.check_output(
                ["git", "rev-parse", "--is-inside-work-tree"],  # noqa: S607
                cwd=self.path,
                encoding="utf-8",
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
