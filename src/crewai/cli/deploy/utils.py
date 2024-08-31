import sys
import re
import subprocess

from rich.console import Console

from ..authentication.utils import TokenManager

console = Console()


if sys.version_info >= (3, 11):
    import tomllib


# Drop the simple_toml_parser when we move to python3.11
def simple_toml_parser(content):
    result = {}
    current_section = result
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            # New section
            section = line[1:-1].split('.')
            current_section = result
            for key in section:
                current_section = current_section.setdefault(key, {})
        elif '=' in line:
            key, value = line.split('=', 1)
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
        console.print(f"Error running trying to fetch the Git Repository: {e}", style="bold red")
    except FileNotFoundError:
        console.print("Git command not found. Make sure Git is installed and in your PATH.", style="bold red")

    return None


def get_project_name(pyproject_path: str = "pyproject.toml") -> str | None:
    """Get the project name from the pyproject.toml file."""
    try:
        # Read the pyproject.toml file
        with open(pyproject_path, "r") as f:
            pyproject_content = parse_toml(f.read())

        # Extract the project name
        project_name = pyproject_content["tool"]["poetry"]["name"]

        if "crewai" not in pyproject_content["tool"]["poetry"]["dependencies"]:
            raise Exception("crewai is not in the dependencies.")

        return project_name

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

    return None


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
