import os
import re
import subprocess

import tomllib


def get_git_remote_url():
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
            print("No origin remote found.")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error running trying to fetch the Git Repository: {e}")
        return None
    except FileNotFoundError:
        print("Git command not found. Make sure Git is installed and in your PATH.")

        return None


def get_project_name(pyproject_path: str = "pyproject.toml"):
    try:
        # Read the pyproject.toml file
        with open(pyproject_path, "rb") as f:
            pyproject_content = tomllib.load(f)

        # Extract the project name
        project_name = pyproject_content["tool"]["poetry"]["name"]

        return project_name
    except FileNotFoundError:
        print(f"Error: {pyproject_path} not found.")
    except KeyError:
        print("Error: 'name' not found in [tool.poetry] section.")
    except tomllib.TOMLDecodeError:
        print(f"Error: {pyproject_path} is not a valid TOML file.")

    return None


def fetch_and_json_env_file(env_file_path: str = ".env") -> dict:
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


def get_auth_token():
    return os.environ.get(
        "TOKEN", "958303356b9a21884a83ddb6e774cc06c6f1dd0e04222fbc5a4e8a9ae02c140e"
    )
