import os
import shutil
import click
import sys
import importlib.metadata

from crewai.cli.authentication.utils import TokenManager
from functools import reduce
from rich.console import Console
from typing import Any, Dict, List

if sys.version_info >= (3, 11):
    import tomllib

console = Console()


def copy_template(src, dst, name, class_name, folder_name):
    """Copy a file from src to dst."""
    with open(src, "r") as file:
        content = file.read()

    # Interpolate the content
    content = content.replace("{{name}}", name)
    content = content.replace("{{crew_name}}", class_name)
    content = content.replace("{{folder_name}}", folder_name)

    # Write the interpolated content to the new file
    with open(dst, "w") as file:
        file.write(content)

    click.secho(f"  - Created {dst}", fg="green")


# Drop the simple_toml_parser when we move to python3.11
def simple_toml_parser(content):
    result = {}
    current_section = result
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            # New section
            section = line[1:-1].split(".")
            current_section = result
            for key in section:
                current_section = current_section.setdefault(key, {})
        elif "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"')
            current_section[key] = value
    return result


def parse_toml(content):
    if sys.version_info >= (3, 11):
        return tomllib.loads(content)
    else:
        return simple_toml_parser(content)


def get_project_name(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Get the project name from the pyproject.toml file."""
    return _get_project_attribute(
        pyproject_path, ["tool", "poetry", "name"], require=require
    )


def get_project_version(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Get the project version from the pyproject.toml file."""
    return _get_project_attribute(
        pyproject_path, ["tool", "poetry", "version"], require=require
    )


def get_project_description(
    pyproject_path: str = "pyproject.toml", require: bool = False
) -> str | None:
    """Get the project description from the pyproject.toml file."""
    return _get_project_attribute(
        pyproject_path, ["tool", "poetry", "description"], require=require
    )


def _get_project_attribute(
    pyproject_path: str, keys: List[str], require: bool
) -> Any | None:
    """Get an attribute from the pyproject.toml file."""
    attribute = None

    try:
        with open(pyproject_path, "r") as f:
            pyproject_content = parse_toml(f.read())

        dependencies = (
            _get_nested_value(pyproject_content, ["tool", "poetry", "dependencies"])
            or {}
        )
        if "crewai" not in dependencies:
            raise Exception("crewai is not in the dependencies.")

        attribute = _get_nested_value(pyproject_content, keys)
    except FileNotFoundError:
        print(f"Error: {pyproject_path} not found.")
    except KeyError:
        print(f"Error: {pyproject_path} is not a valid pyproject.toml file.")
    except tomllib.TOMLDecodeError if sys.version_info >= (3, 11) else Exception as e:  # type: ignore
        print(
            f"Error: {pyproject_path} is not a valid TOML file."
            if sys.version_info >= (3, 11)
            else f"Error reading the pyproject.toml file: {e}"
        )
    except Exception as e:
        print(f"Error reading the pyproject.toml file: {e}")

    if require and not attribute:
        console.print(
            f"Unable to read '{'.'.join(keys)}' in the pyproject.toml file. Please verify that the file exists and contains the specified attribute.",
            style="bold red",
        )
        raise SystemExit

    return attribute


def _get_nested_value(data: Dict[str, Any], keys: List[str]) -> Any:
    return reduce(dict.__getitem__, keys, data)


def get_crewai_version() -> str:
    """Get the version number of CrewAI running the CLI"""
    return importlib.metadata.version("crewai")


def fetch_and_json_env_file(env_file_path: str = ".env") -> dict:
    """Fetch the environment variables from a .env file and return them as a dictionary."""
    try:
        # Read the .env file
        with open(env_file_path, "r") as f:
            env_content = f.read()

        # Parse the .env file content to a dictionary
        env_dict = {}
        for line in env_content.splitlines():
            if line.strip() and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env_dict[key.strip()] = value.strip()

        return env_dict

    except FileNotFoundError:
        print(f"Error: {env_file_path} not found.")
    except Exception as e:
        print(f"Error reading the .env file: {e}")

    return {}


def get_auth_token() -> str:
    """Get the authentication token."""
    access_token = TokenManager().get_token()
    if not access_token:
        raise Exception()
    return access_token


def tree_copy(source, destination):
    """Copies the entire directory structure from the source to the destination."""
    for item in os.listdir(source):
        source_item = os.path.join(source, item)
        destination_item = os.path.join(destination, item)
        if os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item)
        else:
            shutil.copy2(source_item, destination_item)


def tree_find_and_replace(directory, find, replace):
    """Recursively searches through a directory, replacing a target string in
    both file contents and filenames with a specified replacement string.
    """
    for path, dirs, files in os.walk(os.path.abspath(directory), topdown=False):
        for filename in files:
            filepath = os.path.join(path, filename)

            with open(filepath, "r") as file:
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
