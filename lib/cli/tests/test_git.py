import pytest
from crewai_cli.git import Repository


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


def test_fetch_failure_raises_value_error(fp):
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "fetch"], returncode=128, stderr="remote unavailable\n")

    with pytest.raises(
        ValueError,
        match=r"Git fetch failed with exit code 128 for command \['git', 'fetch'\]: remote unavailable",
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


def test_initialize_creates_initial_commit(fp, tmp_path):
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "init"], stdout="")
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    fp.register(["git", "rev-parse", "--verify", "HEAD"], returncode=1)
    fp.register(["git", "add", "."], stdout="")
    fp.register(
        [
            "git",
            "-c",
            "user.name=CrewAI",
            "-c",
            "user.email=deploy@crewai.com",
            "commit",
            "--allow-empty",
            "-m",
            "Initial crew",
        ],
        stdout="",
    )

    repo = Repository.initialize(path=str(tmp_path))

    assert repo.path == str(tmp_path)
    exclude_file = tmp_path / ".git" / "info" / "exclude"
    exclude_text = exclude_file.read_text()
    assert ".env" in exclude_text
    assert "!.env.example" in exclude_text
    assert "!.env.sample" in exclude_text
    assert "__pycache__/" in exclude_text


def test_deployable_files_uses_git_excludes(fp, repository):
    fp.register(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        stdout="pyproject.toml\nsrc/main.py\n",
    )

    assert repository.deployable_files() == ["pyproject.toml", "src/main.py"]


def test_ensure_initial_commit_excludes_preserves_utf8_content(fp, tmp_path):
    """Existing UTF-8 exclude patterns are kept byte-for-byte on every platform.

    Git stores ignore patterns as UTF-8 bytes. Reading them with the locale
    encoding raised ``UnicodeDecodeError`` on Windows (cp1252) for characters
    such as curly quotes, and rewriting them could convert line endings.
    """
    fp.register(["git", "--version"], stdout="git version 2.30.0\n")
    fp.register(["git", "rev-parse", "--is-inside-work-tree"], stdout="true\n")
    exclude_file = tmp_path / ".git" / "info" / "exclude"
    exclude_file.parent.mkdir(parents=True)
    original = "notes \u201cdraft\u201d.md\nb\u0142\u0119dy/\n".encode("utf-8")
    exclude_file.write_bytes(original)

    Repository(path=str(tmp_path), fetch=False)._ensure_initial_commit_excludes()

    written = exclude_file.read_bytes()
    assert written.startswith(original)
    assert b"\r\n" not in written
    text = written.decode("utf-8")
    assert "# CrewAI deploy auto-commit excludes" in text
    assert ".env" in text.splitlines()
    assert "__pycache__/" in text.splitlines()
