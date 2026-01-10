import gc
import weakref

import pytest

from crewai.cli.git import Repository


@pytest.fixture()
def repository(fp):
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")
    return Repository(path=".")


def test_init_with_invalid_git_repo(fp):
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(
        ["git", "rev-parse", "--is-inside-work-tree"],
        returncode=1,
        stderr="fatal: not a git repository\n",
    )

    with pytest.raises(ValueError):
        Repository(path="invalid/path")


def test_is_git_not_installed(fp):
    fp.register(["git", "--version"], returncode=1)

    with pytest.raises(
        ValueError, match="Git is not installed or not found in your PATH."
    ):
        Repository(path=".")


def test_status(fp, repository):
    fp.register(
        ["git", "status", "--branch", "--porcelain"],
        stdout="## main...origin/main [ahead 1]\n",
    )
    assert repository.status() == "## main...origin/main [ahead 1]"


def test_has_uncommitted_changes(fp, repository):
    fp.register(
        ["git", "status", "--branch", "--porcelain"],
        stdout="## main...origin/main\n M somefile.txt\n",
    )
    assert repository.has_uncommitted_changes() is True


def test_is_ahead_or_behind(fp, repository):
    fp.register(
        ["git", "status", "--branch", "--porcelain"],
        stdout="## main...origin/main [ahead 1]\n",
    )
    assert repository.is_ahead_or_behind() is True


def test_is_synced_when_synced(fp, repository):
    fp.register(
        ["git", "status", "--branch", "--porcelain"], stdout="## main...origin/main\n"
    )
    fp.register(
        ["git", "status", "--branch", "--porcelain"], stdout="## main...origin/main\n"
    )
    assert repository.is_synced() is True


def test_is_synced_with_uncommitted_changes(fp, repository):
    fp.register(
        ["git", "status", "--branch", "--porcelain"],
        stdout="## main...origin/main\n M somefile.txt\n",
    )
    assert repository.is_synced() is False


def test_is_synced_when_ahead_or_behind(fp, repository):
    fp.register(
        ["git", "status", "--branch", "--porcelain"],
        stdout="## main...origin/main [ahead 1]\n",
    )
    fp.register(
        ["git", "status", "--branch", "--porcelain"],
        stdout="## main...origin/main [ahead 1]\n",
    )
    assert repository.is_synced() is False


def test_is_synced_with_uncommitted_changes_and_ahead(fp, repository):
    fp.register(
        ["git", "status", "--branch", "--porcelain"],
        stdout="## main...origin/main [ahead 1]\n M somefile.txt\n",
    )
    assert repository.is_synced() is False


def test_origin_url(fp, repository):
    fp.register(
        ["git", "remote", "get-url", "origin"],
        stdout="https://github.com/user/repo.git\n",
    )
    assert repository.origin_url() == "https://github.com/user/repo.git"


def test_repository_instances_are_garbage_collected(fp):
    """Test that Repository instances don't leak memory after going out of scope."""
    # Register git commands
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    # Track weak references
    refs = []

    # Create multiple instances
    for i in range(100):
        repo = Repository(path=".")
        refs.append(weakref.ref(repo))
        del repo

    # Force garbage collection
    gc.collect()

    # Check that instances were garbage collected
    alive = sum(1 for ref in refs if ref() is not None)
    assert alive == 0, f"{alive} Repository instances still alive after gc"


def test_is_git_repo_caches_result(fp):
    """Test that is_git_repo caches its result per instance."""
    # Register git commands
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    repo = Repository(path=".")

    # Cache should be populated after __init__ calls is_git_repo
    assert repo._is_git_repo_cache is True

    # Calling again should use cache (no additional subprocess calls)
    result = repo.is_git_repo()
    assert result is True

    # Verify cache is being used
    assert repo._is_git_repo_cache is True


def test_cache_is_per_instance(fp):
    """Test that cache is per-instance, not global."""
    # Register git commands
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    repo1 = Repository(path=".")
    repo2 = Repository(path=".")

    # Each instance should have its own cache
    assert repo1._is_git_repo_cache is True
    assert repo2._is_git_repo_cache is True

    # Modifying one shouldn't affect the other
    repo1._is_git_repo_cache = False
    assert repo2._is_git_repo_cache is True
