from __future__ import annotations

from functools import reduce
import os
from pathlib import Path
import shutil
import sys
from typing import Any, cast

import click
from rich.console import Console
import tomli

from crewai_cli.config import Settings
from crewai_cli.constants import ENV_VARS


if sys.version_info >= (3, 11):
    import tomllib

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


def read_toml(file_path: str = "pyproject.toml") -> dict[str, Any]:
    """Read the content of a TOML file and return it as a dictionary."""
    with open(file_path, "rb") as f:
        return tomli.load(f)


def parse_toml(content: str) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        return tomllib.loads(content)
    return tomli.loads(content)


def get_project_name(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Get the project name from the pyproject.toml file."""
    return _get_project_attribute(pyproject_path, ["project", "name"], require=require)


def get_project_version(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Get the project version from the pyproject.toml file."""
    return _get_project_attribute(
        pyproject_path, ["project", "version"], require=require
    )


def get_project_description(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Get the project description from the pyproject.toml file."""
    return _get_project_attribute(
        pyproject_path, ["project", "description"], require=require
    )


def _get_project_attribute(
    pyproject_path: str, keys: list[str], require: bool
) -> Any | None:
    """Get an attribute from the pyproject.toml file."""
    attribute = None

    try:
        with open(pyproject_path, "r") as f:
            pyproject_content = parse_toml(f.read())

        dependencies = (
            _get_nested_value(pyproject_content, ["project", "dependencies"]) or []
        )
        if not any(True for dep in dependencies if "crewai" in dep):
            raise Exception("crewai is not in the dependencies.")

        attribute = _get_nested_value(pyproject_content, keys)
    except FileNotFoundError:
        console.print(f"Error: {pyproject_path} not found.", style="bold red")
    except KeyError:
        console.print(
            f"Error: {pyproject_path} is not a valid pyproject.toml file.",
            style="bold red",
        )
    except Exception as e:
        if sys.version_info >= (3, 11) and isinstance(e, tomllib.TOMLDecodeError):
            console.print(
                f"Error: {pyproject_path} is not a valid TOML file.", style="bold red"
            )
        else:
            console.print(
                f"Error reading the pyproject.toml file: {e}", style="bold red"
            )

    if require and not attribute:
        console.print(
            f"Unable to read '{'.'.join(keys)}' in the pyproject.toml file. Please verify that the file exists and contains the specified attribute.",
            style="bold red",
        )
        raise SystemExit

    return attribute


def _get_nested_value(data: dict[str, Any], keys: list[str]) -> Any:
    return reduce(dict.__getitem__, keys, data)


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


def update_env_vars(
    env_vars: dict[str, Any], provider: str, model: str
) -> dict[str, Any] | None:
    """Updates environment variables with the API key for the selected provider and model."""
    provider_config = cast(
        list[str],
        ENV_VARS.get(
            provider,
            [
                click.prompt(
                    f"Enter the environment variable name for your {provider.capitalize()} API key",
                    type=str,
                )
            ],
        ),
    )

    api_key_var = provider_config[0]

    if api_key_var not in env_vars:
        try:
            env_vars[api_key_var] = click.prompt(
                f"Enter your {provider.capitalize()} API key", type=str, hide_input=True
            )
        except click.exceptions.Abort:
            click.secho("Operation aborted by the user.", fg="red")
            return None
    else:
        click.secho(f"API key already exists for {provider.capitalize()}.", fg="yellow")

    env_vars["MODEL"] = model
    click.secho(f"Selected model: {model}", fg="green")
    return env_vars


def write_env_file(folder_path: Path, env_vars: dict[str, Any]) -> None:
    """Writes environment variables to a .env file in the specified folder."""
    env_file_path = folder_path / ".env"
    with open(env_file_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key.upper()}={value}\n")


def build_env_with_tool_repository_credentials(
    repository_handle: str,
) -> dict[str, Any]:
    repository_handle = repository_handle.upper().replace("-", "_")
    settings = Settings()

    env = os.environ.copy()
    env[f"UV_INDEX_{repository_handle}_USERNAME"] = str(
        settings.tool_repository_username or ""
    )
    env[f"UV_INDEX_{repository_handle}_PASSWORD"] = str(
        settings.tool_repository_password or ""
    )

    return env


def build_env_with_all_tool_credentials() -> dict[str, Any]:
    """Build environment dict with credentials for all tool repository indexes.

    Reads ``[tool.uv.sources]`` from ``pyproject.toml`` and merges credentials
    for each private index into a copy of the current environment.

    Returns:
        Environment variables with credentials for all private indexes.
    """
    env = os.environ.copy()
    try:
        pyproject_data = read_toml()
        sources = pyproject_data.get("tool", {}).get("uv", {}).get("sources", {})

        for source_config in sources.values():
            if isinstance(source_config, dict):
                index = source_config.get("index")
                if index:
                    index_env = build_env_with_tool_repository_credentials(index)
                    env.update(index_env)
    except Exception:  # noqa: S110
        pass

    return env
