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


# ============================================================================
# Memory Leak Tests for Issue #4210
# ============================================================================
# These tests verify that Repository instances are properly garbage collected
# after the fix that replaced @lru_cache with instance-level caching.


import gc
import weakref


def test_repository_can_be_garbage_collected(fp):
    """Verify Repository instances are garbage collected after going out of scope.

    This test ensures the fix for issue #4210 works correctly. The @lru_cache
    decorator was causing memory leaks because it held references to `self`,
    preventing garbage collection.
    """
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    repo = Repository(path=".")
    weak_ref = weakref.ref(repo)

    # Call is_git_repo to trigger caching
    assert repo.is_git_repo() is True
    # Call again to verify cache works
    assert repo.is_git_repo() is True

    # Delete the repository instance
    del repo

    # Force garbage collection
    gc.collect()

    # The weak reference should be dead (None) if garbage collection worked
    assert weak_ref() is None, (
        "Repository instance was not garbage collected. "
        "This indicates a memory leak, likely due to @lru_cache holding references."
    )


def test_multiple_repositories_are_garbage_collected(fp):
    """Verify multiple Repository instances are all properly garbage collected.

    This test simulates the scenario described in issue #4210 where
    multiple Repository instances are created in a loop.
    """
    weak_refs = []

    # Create multiple Repository instances
    for i in range(5):
        fp.register(["git", "--version"], stdout="git version 2.30.0\n")
        fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
        fp.register(["git", "fetch"], stdout="")

        repo = Repository(path=".")
        # Use the cached method
        repo.is_git_repo()
        weak_refs.append(weakref.ref(repo))
        del repo

    # Force garbage collection
    gc.collect()

    # All instances should be garbage collected
    collected = sum(1 for ref in weak_refs if ref() is None)
    assert collected == 5, (
        f"Only {collected}/5 Repository instances were garbage collected. "
        "Memory leak detected."
    )


def test_is_git_repo_caching_works(fp):
    """Verify is_git_repo() still caches results correctly with instance-level caching."""
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    repo = Repository(path=".")

    # First call populates cache
    result1 = repo.is_git_repo()
    assert result1 is True

    # Verify cache attribute is set
    assert repo._is_git_repo_cache is True

    # Second call should use cache
    result2 = repo.is_git_repo()
    assert result2 is True

    # Results should be consistent
    assert result1 == result2
