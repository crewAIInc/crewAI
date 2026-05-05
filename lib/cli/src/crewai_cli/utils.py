from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Any

import click
from crewai_core.project import (
    get_project_description as get_project_description,
    get_project_name as get_project_name,
    get_project_version as get_project_version,
    parse_toml as parse_toml,
    read_toml as read_toml,
)
from crewai_core.tool_credentials import (
    build_env_with_all_tool_credentials as build_env_with_all_tool_credentials,
    build_env_with_tool_repository_credentials as build_env_with_tool_repository_credentials,
)
from rich.console import Console


__all__ = [
    "build_env_with_all_tool_credentials",
    "build_env_with_tool_repository_credentials",
    "copy_template",
    "fetch_and_json_env_file",
    "get_project_description",
    "get_project_name",
    "get_project_version",
    "load_env_vars",
    "parse_toml",
    "read_toml",
    "tree_copy",
    "tree_find_and_replace",
    "write_env_file",
]


console = Console()


def copy_template(
    src: Path, dst: Path, name: str, class_name: str, folder_name: str
) -> None:
    """Copy a file from src to dst."""
    with open(src, "r") as file:
        content = file.read()

    content = content.replace("{{name}}", name)
    content = content.replace("{{crew_name}}", class_name)
    content = content.replace("{{folder_name}}", folder_name)

    with open(dst, "w") as file:
        file.write(content)

    click.secho(f"  - Created {dst}", fg="green")


def fetch_and_json_env_file(env_file_path: str = ".env") -> dict[str, Any]:
    """Fetch the environment variables from a .env file and return them as a dictionary."""
    try:
        with open(env_file_path, "r") as f:
            env_content = f.read()

        env_dict = {}
        for line in env_content.splitlines():
            if line.strip() and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env_dict[key.strip()] = value.strip()

        return env_dict

    except FileNotFoundError:
        console.print(f"Error: {env_file_path} not found.", style="bold red")
    except Exception as e:
        console.print(f"Error reading the .env file: {e}", style="bold red")

    return {}


def tree_copy(source: Path, destination: Path) -> None:
    """Copies the entire directory structure from the source to the destination."""
    for item in os.listdir(source):
        source_item = os.path.join(source, item)
        destination_item = os.path.join(destination, item)
        if os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item)
        else:
            shutil.copy2(source_item, destination_item)


def tree_find_and_replace(directory: Path, find: str, replace: str) -> None:
    """Recursively searches through a directory, replacing a target string in
    both file contents and filenames with a specified replacement string.
    """
    for path, dirs, files in os.walk(os.path.abspath(directory), topdown=False):
        for filename in files:
            filepath = os.path.join(path, filename)

            with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
                contents = file.read()
            with open(filepath, "w") as file:
                file.write(contents.replace(find, replace))

            if find in filename:
                new_filename = filename.replace(find, replace)
                new_filepath = os.path.join(path, new_filename)
                os.rename(filepath, new_filepath)

        for dirname in dirs:
            if find in dirname:
                new_dirname = dirname.replace(find, replace)
                new_dirpath = os.path.join(path, new_dirname)
                old_dirpath = os.path.join(path, dirname)
                os.rename(old_dirpath, new_dirpath)


def load_env_vars(folder_path: Path) -> dict[str, Any]:
    """Loads environment variables from a .env file in the specified folder path."""
    env_file_path = folder_path / ".env"
    env_vars = {}
    if env_file_path.exists():
        with open(env_file_path, "r") as file:
            for line in file:
                key, _, value = line.strip().partition("=")
                if key and value:
                    env_vars[key] = value
    return env_vars


def write_env_file(folder_path: Path, env_vars: dict[str, Any]) -> None:
    """Writes environment variables to a .env file in the specified folder."""
    env_file_path = folder_path / ".env"
    with open(env_file_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key.upper()}={value}\n")
