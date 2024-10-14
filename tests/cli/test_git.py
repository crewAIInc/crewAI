from crewai.cli.git import Repository
import pytest


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
