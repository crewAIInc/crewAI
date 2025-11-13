from functools import reduce
import importlib.util
from inspect import getmro, isclass, isfunction, ismethod
import os
from pathlib import Path
import shutil
import sys
from typing import Any, cast, get_type_hints

import click
from rich.console import Console
import tomli

from crewai.cli.config import Settings
from crewai.cli.constants import ENV_VARS
from crewai.crew import Crew
from crewai.flow import Flow


if sys.version_info >= (3, 11):
    import tomllib

console = Console()


def copy_template(
    src: Path, dst: Path, name: str, class_name: str, folder_name: str
) -> None:
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
        # Handle TOML decode errors for Python 3.11+
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


def update_env_vars(
    env_vars: dict[str, Any], provider: str, model: str
) -> dict[str, Any] | None:
    """
    Updates environment variables with the API key for the selected provider and model.

    Args:
    - env_vars (dict): Environment variables dictionary.
    - provider (str): Selected provider.
    - model (str): Selected model.

    Returns:
    - None
    """
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
    """
    Writes environment variables to a .env file in the specified folder.

    Args:
    - folder_path (Path): The path to the folder where the .env file will be written.
    - env_vars (dict): A dictionary of environment variables to write.
    """
    env_file_path = folder_path / ".env"
    with open(env_file_path, "w") as file:
        for key, value in env_vars.items():
            file.write(f"{key.upper()}={value}\n")


def get_crews(crew_path: str = "crew.py", require: bool = False) -> list[Crew]:
    """Get the crew instances from a file."""
    crew_instances = []
    try:
        import importlib.util

        # Add the current directory to sys.path to ensure imports resolve correctly
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)

        # If we're not in src directory but there's a src directory, add it to path
        src_dir = os.path.join(current_dir, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

        # Search in both current directory and src directory if it exists
        search_paths = [".", "src"] if os.path.isdir("src") else ["."]

        for search_path in search_paths:
            for root, _, files in os.walk(search_path):
                if crew_path in files and "cli/templates" not in root:
                    crew_os_path = os.path.join(root, crew_path)
                    try:
                        spec = importlib.util.spec_from_file_location(
                            "crew_module", crew_os_path
                        )
                        if not spec or not spec.loader:
                            continue

                        module = importlib.util.module_from_spec(spec)
                        sys.modules[spec.name] = module

                        try:
                            spec.loader.exec_module(module)

                            for attr_name in dir(module):
                                module_attr = getattr(module, attr_name)
                                try:
                                    crew_instances.extend(fetch_crews(module_attr))
                                except Exception as e:
                                    console.print(
                                        f"Error processing attribute {attr_name}: {e}",
                                        style="bold red",
                                    )
                                    continue

                            # If we found crew instances, break out of the loop
                            if crew_instances:
                                break

                        except Exception as exec_error:
                            console.print(
                                f"Error executing module: {exec_error}",
                                style="bold red",
                            )

                    except (ImportError, AttributeError) as e:
                        if require:
                            console.print(
                                f"Error importing crew from {crew_path}: {e!s}",
                                style="bold red",
                            )
                        continue

            # If we found crew instances in this search path, break out of the search paths loop
            if crew_instances:
                break

        if require and not crew_instances:
            console.print("No valid Crew instance found in crew.py", style="bold red")
            raise SystemExit

    except Exception as e:
        if require:
            console.print(
                f"Unexpected error while loading crew: {e!s}", style="bold red"
            )
            raise SystemExit from e
    return crew_instances


def get_crew_instance(module_attr: Any) -> Crew | None:
    if (
        callable(module_attr)
        and hasattr(module_attr, "is_crew_class")
        and module_attr.is_crew_class
    ):
        return cast(Crew, module_attr().crew())
    try:
        if (ismethod(module_attr) or isfunction(module_attr)) and get_type_hints(
            module_attr
        ).get("return") is Crew:
            return cast(Crew, module_attr())
    except Exception:
        return None

    if isinstance(module_attr, Crew):
        return module_attr
    return None


def fetch_crews(module_attr: Any) -> list[Crew]:
    crew_instances: list[Crew] = []

    if crew_instance := get_crew_instance(module_attr):
        crew_instances.append(crew_instance)

    if isinstance(module_attr, type) and issubclass(module_attr, Flow):
        instance = module_attr()
        for attr_name in dir(instance):
            attr = getattr(instance, attr_name)
            if crew_instance := get_crew_instance(attr):
                crew_instances.append(crew_instance)
    return crew_instances


def is_valid_tool(obj: Any) -> bool:
    from crewai.tools.base_tool import Tool

    if isclass(obj):
        try:
            return any(base.__name__ == "BaseTool" for base in getmro(obj))
        except (TypeError, AttributeError):
            return False

    return isinstance(obj, Tool)


def extract_available_exports(dir_path: str = "src") -> list[dict[str, Any]]:
    """
    Extract available tool classes from the project's __init__.py files.
    Only includes classes that inherit from BaseTool or functions decorated with @tool.

    Returns:
        list: A list of valid tool class names or ["BaseTool"] if none found
    """
    try:
        init_files = Path(dir_path).glob("**/__init__.py")
        available_exports = []

        for init_file in init_files:
            tools = _load_tools_from_init(init_file)
            available_exports.extend(tools)

        if not available_exports:
            _print_no_tools_warning()
            raise SystemExit(1)

        return available_exports

    except Exception as e:
        console.print(f"[red]Error: Could not extract tool classes: {e!s}[/red]")
        console.print(
            "Please ensure your project contains valid tools (classes inheriting from BaseTool or functions with @tool decorator)."
        )
        raise SystemExit(1) from e


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


def _load_tools_from_init(init_file: Path) -> list[dict[str, Any]]:
    """
    Load and validate tools from a given __init__.py file.
    """
    spec = importlib.util.spec_from_file_location("temp_module", init_file)

    if not spec or not spec.loader:
        return []

    module = importlib.util.module_from_spec(spec)
    sys.modules["temp_module"] = module

    try:
        spec.loader.exec_module(module)

        if not hasattr(module, "__all__"):
            console.print(
                f"Warning: No __all__ defined in {init_file}",
                style="bold yellow",
            )
            raise SystemExit(1)

        return [
            {
                "name": name,
            }
            for name in module.__all__
            if hasattr(module, name) and is_valid_tool(getattr(module, name))
        ]

    except Exception as e:
        console.print(f"[red]Warning: Could not load {init_file}: {e!s}[/red]")
        raise SystemExit(1) from e

    finally:
        sys.modules.pop("temp_module", None)


def _print_no_tools_warning() -> None:
    """
    Display warning and usage instructions if no tools were found.
    """
    console.print(
        "\n[bold yellow]Warning: No valid tools were exposed in your __init__.py file![/bold yellow]"
    )
    console.print(
        "Your __init__.py file must contain all classes that inherit from [bold]BaseTool[/bold] "
        "or functions decorated with [bold]@tool[/bold]."
    )
    console.print(
        "\nExample:\n[dim]# In your __init__.py file[/dim]\n"
        "[green]__all__ = ['YourTool', 'your_tool_function'][/green]\n\n"
        "[dim]# In your tool.py file[/dim]\n"
        "[green]from crewai.tools import BaseTool, tool\n\n"
        "# Tool class example\n"
        "class YourTool(BaseTool):\n"
        '    name = "your_tool"\n'
        '    description = "Your tool description"\n'
        "    # ... rest of implementation\n\n"
        "# Decorated function example\n"
        "@tool\n"
        "def your_tool_function(text: str) -> str:\n"
        '    """Your tool description"""\n'
        "    # ... implementation\n"
        "    return result\n"
    )
