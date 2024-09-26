import click
import re
import subprocess
import sys

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


def get_git_remote_url() -> str | None:
    """Get the Git repository's remote URL."""
    try:
        # Run the git remote -v command
        result = subprocess.run(
            ["git", "remote", "-v"], capture_output=True, text=True, check=True
        )

        # Get the output
        output = result.stdout

        # Parse the output to find the origin URL
        matches = re.findall(r"origin\s+(.*?)\s+\(fetch\)", output)

        if matches:
            return matches[0]  # Return the first match (origin URL)
        else:
            console.print("No origin remote found.", style="bold red")

    except subprocess.CalledProcessError as e:
        console.print(
            f"Error running trying to fetch the Git Repository: {e}", style="bold red"
        )
    except FileNotFoundError:
        console.print(
            "Git command not found. Make sure Git is installed and in your PATH.",
            style="bold red",
        )

    return None


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


def get_crewai_version(poetry_lock_path: str = "poetry.lock") -> str:
    """Get the version number of crewai from the poetry.lock file."""
    try:
        with open(poetry_lock_path, "r") as f:
            lock_content = f.read()

        match = re.search(
            r'\[\[package\]\]\s*name\s*=\s*"crewai"\s*version\s*=\s*"([^"]+)"',
            lock_content,
            re.DOTALL,
        )
        if match:
            return match.group(1)
        else:
            print("crewai package not found in poetry.lock")
            return "no-version-found"

    except FileNotFoundError:
        print(f"Error: {poetry_lock_path} not found.")
    except Exception as e:
        print(f"Error reading the poetry.lock file: {e}")

    return "no-version-found"


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
