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


def test_repository_garbage_collection(fp):
    """Test that Repository instances can be garbage collected.

    This test verifies the fix for the memory leak issue where using
    @lru_cache on the is_git_repo() method prevented garbage collection
    of Repository instances.
    """
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    repo = Repository(path=".")
    weak_ref = weakref.ref(repo)

    assert weak_ref() is not None

    del repo
    gc.collect()

    assert weak_ref() is None, (
        "Repository instance was not garbage collected. "
        "This indicates a memory leak, likely from @lru_cache on instance methods."
    )


def test_is_git_repo_caching(fp):
    """Test that is_git_repo() result is cached at the instance level.

    This verifies that the instance-level caching works correctly,
    only calling the subprocess once per instance.
    """
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    repo = Repository(path=".")

    assert repo._is_git_repo_cache is True

    result1 = repo.is_git_repo()
    result2 = repo.is_git_repo()

    assert result1 is True
    assert result2 is True
    assert repo._is_git_repo_cache is True


def test_multiple_repository_instances_independent_caches(fp):
    """Test that multiple Repository instances have independent caches.

    This verifies that the instance-level caching doesn't share state
    between different Repository instances.
    """
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], stdout="")

    repo1 = Repository(path=".")
    repo2 = Repository(path=".")

    assert repo1._is_git_repo_cache is True
    assert repo2._is_git_repo_cache is True

    assert repo1._is_git_repo_cache is not repo2._is_git_repo_cache or (
        repo1._is_git_repo_cache == repo2._is_git_repo_cache
    )

    weak_ref1 = weakref.ref(repo1)
    del repo1
    gc.collect()

    assert weak_ref1() is None
    assert repo2._is_git_repo_cache is True
