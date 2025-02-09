import os
import shutil
import sys
from functools import reduce
from typing import Any, Dict, List

import click
import tomli
from rich.console import Console

from crewai.cli.constants import ENV_VARS
from crewai.crew import Crew

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


def read_toml(file_path: str = "pyproject.toml"):
    """Read the content of a TOML file and return it as a dictionary."""
    with open(file_path, "rb") as f:
        toml_dict = tomli.load(f)
    return toml_dict


def parse_toml(content):
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
    pyproject_path: str, keys: List[str], require: bool
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


def load_env_vars(folder_path):
    """
    Loads environment variables from a .env file in the specified folder path.

    Args:
    - folder_path (Path): The path to the folder containing the .env file.

    Returns:
    - dict: A dictionary of environment variables.
    """
    env_file_path = folder_path / ".env"
    env_vars = {}
    if env_file_path.exists():
        with open(env_file_path, "r") as file:
            for line in file:
                key, _, value = line.strip().partition("=")
                if key and value:
                    env_vars[key] = value
    return env_vars


def update_env_vars(env_vars, provider, model):
    """
    Updates environment variables with the API key for the selected provider and model.

    Args:
    - env_vars (dict): Environment variables dictionary.
    - provider (str): Selected provider.
    - model (str): Selected model.

    Returns:
    - None
    """
    api_key_var = ENV_VARS.get(
        provider,
        [
            click.prompt(
                f"Enter the environment variable name for your {provider.capitalize()} API key",
                type=str,
            )
        ],
    )[0]

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


def write_env_file(folder_path, env_vars):
    """
    Writes environment variables to a .env file in the specified folder.

    Args:
    - folder_path (Path): The path to the folder where the .env file will be written.
    - env_vars (dict): A dictionary of environment variables to write.
    """
    env_file_path = folder_path / ".env"
    with open(env_file_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key}={value}\n")


def get_crew(crew_path: str = "crew.py", require: bool = False) -> Crew | None:
    """Get the crew instance from the crew.py file."""
    try:
        import importlib.util
        import os

        for root, _, files in os.walk("."):
            if "crew.py" in files:
                crew_path = os.path.join(root, "crew.py")
                try:
                    spec = importlib.util.spec_from_file_location(
                        "crew_module", crew_path
                    )
                    if not spec or not spec.loader:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    try:
                        sys.modules[spec.name] = module
                        spec.loader.exec_module(module)

                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            try:
                                if callable(attr) and hasattr(attr, "crew"):
                                    crew_instance = attr().crew()
                                    return crew_instance

                            except Exception as e:
                                print(f"Error processing attribute {attr_name}: {e}")
                                continue

                    except Exception as exec_error:
                        print(f"Error executing module: {exec_error}")
                        import traceback

                        print(f"Traceback: {traceback.format_exc()}")

                except (ImportError, AttributeError) as e:
                    if require:
                        console.print(
                            f"Error importing crew from {crew_path}: {str(e)}",
                            style="bold red",
                        )
                        continue

                break

        if require:
            console.print("No valid Crew instance found in crew.py", style="bold red")
            raise SystemExit
        return None

    except Exception as e:
        if require:
            console.print(
                f"Unexpected error while loading crew: {str(e)}", style="bold red"
            )
            raise SystemExit
        return None
