"""Code Interpreter Tool for executing Python code in isolated environments.

This module provides a tool for executing Python code in a Docker container for
safe isolation. Docker is required for secure code execution.

SECURITY: This tool executes arbitrary code. Docker isolation is mandatory for
untrusted code. The tool will fail if Docker is not available to prevent
sandbox escape vulnerabilities.
"""

import importlib.util
import os
import subprocess
from typing import Any, TypedDict

from crewai.tools import BaseTool
from docker import (  # type: ignore[import-untyped]
    DockerClient,
    from_env as docker_from_env,
)
from docker.errors import ImageNotFound, NotFound  # type: ignore[import-untyped]
from docker.models.containers import Container  # type: ignore[import-untyped]
from pydantic import BaseModel, Field
from typing_extensions import Unpack

from crewai_tools.printer import Printer


class RunKwargs(TypedDict, total=False):
    """Keyword arguments for the _run method."""

    code: str
    libraries_used: list[str]


class CodeInterpreterSchema(BaseModel):
    """Schema for defining inputs to the CodeInterpreterTool.

    This schema defines the required parameters for code execution,
    including the code to run and any libraries that need to be installed.
    """

    code: str = Field(
        ...,
        description="Python3 code used to be interpreted in the Docker container. ALWAYS PRINT the final result and the output of the code",
    )

    libraries_used: list[str] = Field(
        ...,
        description="List of libraries used in the code with proper installing names separated by commas. Example: numpy,pandas,beautifulsoup4",
    )


class CodeInterpreterTool(BaseTool):
    """A tool for executing Python code in isolated Docker containers.

    This tool provides functionality to run Python code in a Docker container
    for safe isolation. Docker is required for secure code execution.

    Security Model:
    - Docker container provides process, filesystem, and network isolation
    - Code execution fails if Docker is unavailable (fail-safe)
    - unsafe_mode bypasses all protections (use only in trusted environments)

    For more information, see:
    https://docs.crewai.com/en/tools/ai-ml/codeinterpretertool#docker-container-recommended
    """

    name: str = "Code Interpreter"
    description: str = "Interprets Python3 code strings with a final print statement. Requires Docker for secure execution."
    args_schema: type[BaseModel] = CodeInterpreterSchema
    default_image_tag: str = "code-interpreter:latest"
    code: str | None = None
    user_dockerfile_path: str | None = None
    user_docker_base_url: str | None = None
    unsafe_mode: bool = False

    @staticmethod
    def _get_installed_package_path() -> str:
        """Gets the installation path of the crewai_tools package.

        Returns:
            The directory path where the package is installed.

        Raises:
            RuntimeError: If the package cannot be found.
        """
        spec = importlib.util.find_spec("crewai_tools")
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot find crewai_tools package installation path")
        return os.path.dirname(spec.origin)

    def _verify_docker_image(self) -> None:
        """Verifies if the Docker image is available or builds it if necessary.

        Checks if the required Docker image exists. If not, builds it using either a
        user-provided Dockerfile or the default one included with the package.

        Raises:
            FileNotFoundError: If the Dockerfile cannot be found.
        """
        client = (
            docker_from_env()
            if self.user_docker_base_url is None
            else DockerClient(base_url=self.user_docker_base_url)
        )

        try:
            client.images.get(self.default_image_tag)

        except ImageNotFound:
            if self.user_dockerfile_path and os.path.exists(self.user_dockerfile_path):
                dockerfile_path = self.user_dockerfile_path
            else:
                package_path = self._get_installed_package_path()
                dockerfile_path = os.path.join(
                    package_path, "tools/code_interpreter_tool"
                )
                if not os.path.exists(dockerfile_path):
                    raise FileNotFoundError(
                        f"Dockerfile not found in {dockerfile_path}"
                    ) from None

            client.images.build(
                path=dockerfile_path,
                tag=self.default_image_tag,
                rm=True,
            )

    def _run(self, **kwargs: Unpack[RunKwargs]) -> str:
        """Runs the code interpreter tool with the provided arguments.

        Args:
            **kwargs: Keyword arguments that should include 'code' and 'libraries_used'.

        Returns:
            The output of the executed code as a string.
        """
        code: str | None = kwargs.get("code", self.code)
        libraries_used: list[str] = kwargs.get("libraries_used", [])

        if not code:
            return "No code provided to execute."

        if self.unsafe_mode:
            return self.run_code_unsafe(code, libraries_used)
        return self.run_code_safety(code, libraries_used)

    @staticmethod
    def _install_libraries(container: Container, libraries: list[str]) -> None:
        """Installs required Python libraries in the Docker container.

        Args:
            container: The Docker container where libraries will be installed.
            libraries: A list of library names to install using pip.
        """
        for library in libraries:
            container.exec_run(["pip", "install", library])

    def _init_docker_container(self) -> Container:
        """Initializes and returns a Docker container for code execution.

        Stops and removes any existing container with the same name before creating
        a new one. Maps the current working directory to /workspace in the container.

        Returns:
            A Docker container object ready for code execution.
        """
        container_name = "code-interpreter"
        client = docker_from_env()
        current_path = os.getcwd()

        # Check if the container is already running
        try:
            existing_container = client.containers.get(container_name)
            existing_container.stop()
            existing_container.remove()
        except NotFound:
            pass  # Container does not exist, no need to remove

        return client.containers.run(
            self.default_image_tag,
            detach=True,
            tty=True,
            working_dir="/workspace",
            name=container_name,
            volumes={current_path: {"bind": "/workspace", "mode": "rw"}},  # type: ignore
        )

    @staticmethod
    def _check_docker_available() -> bool:
        """Checks if Docker is available and running on the system.

        Attempts to run the 'docker info' command to verify Docker availability.

        Returns:
            True if Docker is available and running, False otherwise.
        """
        try:
            subprocess.run(
                ["docker", "info"],  # noqa: S607
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def run_code_safety(self, code: str, libraries_used: list[str]) -> str:
        """Runs code in a Docker container for safe isolation.

        Requires Docker to be installed and running. Fails with an error message
        if Docker is not available, preventing sandbox escape vulnerabilities.

        Args:
            code: The Python code to execute as a string.
            libraries_used: A list of Python library names to install before execution.

        Returns:
            The output of the executed code as a string, or an error message if
            Docker is not available.

        Raises:
            RuntimeError: If Docker is not available and code execution is attempted.
        """
        if not self._check_docker_available():
            error_msg = (
                "SECURITY ERROR: Docker is required for safe code execution but is not available.\n\n"
                "Docker provides essential isolation to prevent sandbox escape attacks.\n"
                "Please install and start Docker, then try again.\n\n"
                "For installation instructions, see:\n"
                "- https://docs.docker.com/get-docker/\n"
                "- https://docs.crewai.com/en/tools/ai-ml/codeinterpretertool#docker-container-recommended\n\n"
                "If you are in a trusted environment and understand the risks, you can use unsafe_mode=True,\n"
                "but this is NOT recommended for production use or untrusted code."
            )
            Printer.print(error_msg, color="bold_red")
            raise RuntimeError(
                "Docker is required for safe code execution. "
                "Install Docker or use unsafe_mode=True (not recommended)."
            )

        return self.run_code_in_docker(code, libraries_used)

    def run_code_in_docker(self, code: str, libraries_used: list[str]) -> str:
        """Runs Python code in a Docker container for safe isolation.

        Creates a Docker container, installs the required libraries, executes the code,
        and then cleans up by stopping and removing the container.

        Args:
            code: The Python code to execute as a string.
            libraries_used: A list of Python library names to install before execution.

        Returns:
            The output of the executed code as a string, or an error message if execution failed.
        """
        Printer.print("Running code in Docker environment", color="bold_blue")
        self._verify_docker_image()
        container = self._init_docker_container()
        self._install_libraries(container, libraries_used)

        exec_result = container.exec_run(["python3", "-c", code])

        container.stop()
        container.remove()

        if exec_result.exit_code != 0:
            return f"Something went wrong while running the code: \n{exec_result.output.decode('utf-8')}"
        return exec_result.output.decode("utf-8")


    @staticmethod
    def run_code_unsafe(code: str, libraries_used: list[str]) -> str:
        """Runs code directly on the host machine without any safety restrictions.

        WARNING: This mode bypasses all security controls and executes code directly
        on the host system. Use ONLY in trusted environments with trusted code.

        SECURITY RISKS:
        - No process isolation
        - No filesystem restrictions
        - No network restrictions
        - Full access to host system resources
        - Potential for system compromise

        Args:
            code: The Python code to execute as a string.
            libraries_used: A list of Python library names to install before execution.

        Returns:
            The value of the 'result' variable from the executed code,
            or an error message if execution failed.
        """
        Printer.print(
            "⚠️  WARNING: Running code in UNSAFE mode - no security controls active!",
            color="bold_red",
        )

        for library in libraries_used:
            try:
                subprocess.run(
                    ["pip", "install", library],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                return f"Failed to install library '{library}': {e!s}"

        try:
            exec_locals: dict[str, Any] = {}
            exec(code, {}, exec_locals)  # noqa: S102
            return exec_locals.get("result", "No result variable found.")
        except Exception as e:
            return f"An error occurred: {e!s}"
