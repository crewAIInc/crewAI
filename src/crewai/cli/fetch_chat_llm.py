import json
import subprocess

import click
from packaging import version

from crewai.cli.utils import read_toml
from crewai.cli.version import get_crewai_version
from crewai.llm import LLM


def fetch_chat_llm() -> LLM:
    """
    Fetch the chat LLM by running "uv run fetch_chat_llm" (or your chosen script name),
    parsing its JSON stdout, and returning an LLM instance.

    This expects the script "fetch_chat_llm" to print out JSON that represents the
    LLM parameters (e.g., by calling something like: print(json.dumps(llm.to_dict()))).

    Any error, whether from the subprocess or JSON parsing, will raise a RuntimeError.
    """

    # You may change this command to match whatever's in your pyproject.toml [project.scripts].
    command = ["uv", "run", "fetch_chat_llm"]

    crewai_version = get_crewai_version()
    min_required_version = "0.87.0"  # Adjust as needed

    pyproject_data = read_toml()

    # If old poetry-based setup is detected and version is below min_required_version
    if pyproject_data.get("tool", {}).get("poetry") and (
        version.parse(crewai_version) < version.parse(min_required_version)
    ):
        click.secho(
            f"You are running an older version of crewAI ({crewai_version}) that uses poetry pyproject.toml.\n"
            f"Please run `crewai update` to transition your pyproject.toml to use uv.",
            fg="red",
        )

    # Initialize a reference to your LLM
    llm_instance = None

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        stdout_lines = result.stdout.strip().splitlines()

        # Find the line that contains the JSON data
        json_line = next(
            (
                line
                for line in stdout_lines
                if line.startswith("{") and line.endswith("}")
            ),
            None,
        )

        if not json_line:
            raise RuntimeError(
                "No valid JSON output received from `fetch_chat_llm` command."
            )

        try:
            llm_data = json.loads(json_line)
            llm_instance = LLM.from_dict(llm_data)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Unable to parse JSON from `fetch_chat_llm` output: {e}\nOutput: {repr(json_line)}"
            ) from e

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"An error occurred while fetching chat LLM: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while fetching chat LLM: {e}"
        ) from e

    if not llm_instance:
        raise RuntimeError("Failed to create a valid LLM from `fetch_chat_llm` output.")

    return llm_instance
