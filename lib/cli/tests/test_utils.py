import os
import shutil
import tempfile
from pathlib import Path

import click
import pytest
from crewai_cli import utils


@pytest.fixture
def temp_tree():
    root_dir = tempfile.mkdtemp()

    create_file(os.path.join(root_dir, "file1.txt"), "Hello, world!")
    create_file(os.path.join(root_dir, "file2.txt"), "Another file")
    os.mkdir(os.path.join(root_dir, "empty_dir"))
    nested_dir = os.path.join(root_dir, "nested_dir")
    os.mkdir(nested_dir)
    create_file(os.path.join(nested_dir, "nested_file.txt"), "Nested content")

    yield root_dir

    shutil.rmtree(root_dir)


def create_file(path, content):
    with open(path, "w") as f:
        f.write(content)


def test_tree_find_and_replace_file_content(temp_tree):
    utils.tree_find_and_replace(temp_tree, "world", "universe")
    with open(os.path.join(temp_tree, "file1.txt"), "r") as f:
        assert f.read() == "Hello, universe!"


def test_tree_find_and_replace_file_name(temp_tree):
    old_path = os.path.join(temp_tree, "file2.txt")
    new_path = os.path.join(temp_tree, "file2_renamed.txt")
    os.rename(old_path, new_path)
    utils.tree_find_and_replace(temp_tree, "renamed", "modified")
    assert os.path.exists(os.path.join(temp_tree, "file2_modified.txt"))
    assert not os.path.exists(new_path)


def test_tree_find_and_replace_directory_name(temp_tree):
    utils.tree_find_and_replace(temp_tree, "empty", "renamed")
    assert os.path.exists(os.path.join(temp_tree, "renamed_dir"))
    assert not os.path.exists(os.path.join(temp_tree, "empty_dir"))


def test_tree_find_and_replace_nested_content(temp_tree):
    utils.tree_find_and_replace(temp_tree, "Nested", "Updated")
    with open(os.path.join(temp_tree, "nested_dir", "nested_file.txt"), "r") as f:
        assert f.read() == "Updated content"


def test_tree_find_and_replace_no_matches(temp_tree):
    utils.tree_find_and_replace(temp_tree, "nonexistent", "replacement")
    assert set(os.listdir(temp_tree)) == {
        "file1.txt",
        "file2.txt",
        "empty_dir",
        "nested_dir",
    }


def test_tree_copy_full_structure(temp_tree):
    dest_dir = tempfile.mkdtemp()
    try:
        utils.tree_copy(temp_tree, dest_dir)
        assert set(os.listdir(dest_dir)) == set(os.listdir(temp_tree))
        assert os.path.isfile(os.path.join(dest_dir, "file1.txt"))
        assert os.path.isfile(os.path.join(dest_dir, "file2.txt"))
        assert os.path.isdir(os.path.join(dest_dir, "empty_dir"))
        assert os.path.isdir(os.path.join(dest_dir, "nested_dir"))
        assert os.path.isfile(os.path.join(dest_dir, "nested_dir", "nested_file.txt"))
    finally:
        shutil.rmtree(dest_dir)


def test_tree_copy_preserve_content(temp_tree):
    dest_dir = tempfile.mkdtemp()
    try:
        utils.tree_copy(temp_tree, dest_dir)
        with open(os.path.join(dest_dir, "file1.txt"), "r") as f:
            assert f.read() == "Hello, world!"
        with open(os.path.join(dest_dir, "nested_dir", "nested_file.txt"), "r") as f:
            assert f.read() == "Nested content"
    finally:
        shutil.rmtree(dest_dir)


def test_tree_copy_to_existing_directory(temp_tree):
    dest_dir = tempfile.mkdtemp()
    try:
        create_file(os.path.join(dest_dir, "existing_file.txt"), "I was here first")
        utils.tree_copy(temp_tree, dest_dir)
        assert os.path.isfile(os.path.join(dest_dir, "existing_file.txt"))
        assert os.path.isfile(os.path.join(dest_dir, "file1.txt"))
    finally:
        shutil.rmtree(dest_dir)


@pytest.mark.parametrize("value", ["1", "true", "True", "yes", "anything"])
def test_is_dmn_mode_enabled_for_truthy_values(monkeypatch, value):
    monkeypatch.setenv("CREWAI_DMN", value)

    assert utils.is_dmn_mode_enabled() is True


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off"])
def test_is_dmn_mode_enabled_for_falsey_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CREWAI_DMN", raising=False)
    else:
        monkeypatch.setenv("CREWAI_DMN", value)

    assert utils.is_dmn_mode_enabled() is False


def test_project_id_lookups_default_to_the_cwd_pyproject(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[tool.crewai]\nproject_id = 'proj-1'\n")

    assert utils.get_project_id() == "proj-1"
    assert utils.get_or_create_project_id() == "proj-1"


# Tests for extract_available_exports, get_crews, get_flows, fetch_crews,
# is_valid_tool live in lib/crewai/tests/cli/test_utils.py — the canonical
# implementations are in crewai.utilities.project_utils.


@pytest.mark.usefixtures("no_provider_keys")
@pytest.mark.parametrize(
    ("provider", "env_vars", "expected"),
    [
        ("openai", {}, ["OPENAI_API_KEY"]),
        ("openai", {"OPENAI_API_KEY": ""}, ["OPENAI_API_KEY"]),
        ("openai", {"OPENAI_API_KEY": "sk-test"}, []),
        ("bedrock", {"aws_access_key_id": "AKIATEST"}, ["AWS_SECRET_ACCESS_KEY"]),
        ("azure", {}, ["AZURE_API_KEY", "AZURE_API_BASE"]),
        ("watson", {}, ["WATSONX_URL", "WATSONX_APIKEY", "WATSONX_PROJECT_ID"]),
        ("ollama", {}, []),
        ("deepseek", {}, []),
    ],
)
def test_missing_provider_keys_names_each_required_value_left_unset(
    provider, env_vars, expected
):
    assert utils.missing_provider_keys(provider, env_vars) == expected


@pytest.mark.usefixtures("no_provider_keys")
def test_missing_provider_keys_skips_a_key_set_in_the_environment(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-from-shell")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "")

    assert utils.missing_provider_keys("bedrock", {}) == ["AWS_SECRET_ACCESS_KEY"]


owner_only_files = pytest.mark.skipif(
    os.name == "nt", reason="POSIX permission bits"
)


@owner_only_files
def test_write_env_file_creates_an_env_only_its_owner_can_read(tmp_path):
    old_umask = os.umask(0o022)
    try:
        utils.write_env_file(tmp_path, {"openai_api_key": "sk-test"})
    finally:
        os.umask(old_umask)

    env_file = tmp_path / ".env"
    assert env_file.read_text() == "OPENAI_API_KEY=sk-test\n"
    assert env_file.stat().st_mode & 0o777 == 0o600


@owner_only_files
def test_write_env_file_makes_an_existing_readable_env_owner_only(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OLD=1\n")
    env_file.chmod(0o644)

    utils.write_env_file(tmp_path, {"MODEL": "openai/gpt-6-luna"})

    assert env_file.read_text() == "MODEL=openai/gpt-6-luna\n"
    assert env_file.stat().st_mode & 0o777 == 0o600


def test_write_env_file_never_writes_keys_through_a_symlink(tmp_path):
    target = tmp_path / "elsewhere"
    target.write_text("untouched\n")
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").symlink_to(target)

    with pytest.raises(click.ClickException, match="it is a symlink"):
        utils.write_env_file(project, {"OPENAI_API_KEY": "sk-test"})

    assert target.read_text() == "untouched\n"
