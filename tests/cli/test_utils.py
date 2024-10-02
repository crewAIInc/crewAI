import pytest
import shutil
import tempfile
import os
from crewai.cli import utils


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
