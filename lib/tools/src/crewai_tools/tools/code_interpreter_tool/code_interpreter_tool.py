"""Code Interpreter Tool for executing Python code in isolated environments.

This module provides a tool for executing Python code either in a Docker container for
safe isolation or directly in a restricted sandbox. It includes mechanisms for blocking
potentially unsafe operations and importing restricted modules.
"""

import importlib.util
import os
from types import ModuleType
from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from docker import DockerClient
from docker import from_env as docker_from_env
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container
from pydantic import BaseModel, Field

from crewai_tools.printer import Printer


class CodeInterpreterSchema(BaseModel):
    """Schema for defining inputs to the CodeInterpreterTool.

    This schema defines the required parameters for code execution,
    including the code to run and any libraries that need to be installed.
    """

    code: str = Field(
        ...,
        description="Python3 code used to be interpreted in the Docker container. ALWAYS PRINT the final result and the output of the code",
    )

    libraries_used: List[str] = Field(
        ...,
        description="List of libraries used in the code with proper installing names separated by commas. Example: numpy,pandas,beautifulsoup4",
    )


class SandboxPython:
    """A restricted Python execution environment for running code safely.

    This class provides methods to safely execute Python code by restricting access to
    potentially dangerous modules and built-in functions. It creates a sandboxed
    environment where harmful operations are blocked.
    """

    BLOCKED_MODULES = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "importlib",
        "inspect",
        "tempfile",
        "sysconfig",
        "builtins",
    }

    UNSAFE_BUILTINS = {
        "exec",
        "eval",
        "open",
        "compile",
        "input",
        "globals",
        "locals",
        "vars",
        "help",
        "dir",
    }

    @staticmethod
    def restricted_import(
        name: str,
        custom_globals: Optional[Dict[str, Any]] = None,
        custom_locals: Optional[Dict[str, Any]] = None,
        fromlist: Optional[List[str]] = None,
        level: int = 0,
    ) -> ModuleType:
        """A restricted import function that blocks importing of unsafe modules.

        Args:
            name: The name of the module to import.
            custom_globals: Global namespace to use.
            custom_locals: Local namespace to use.
            fromlist: List of items to import from the module.
            level: The level value passed to __import__.

        Returns:
            The imported module if allowed.

        Raises:
            ImportError: If the module is in the blocked modules list.
        """
        if name in SandboxPython.BLOCKED_MODULES:
            raise ImportError(f"Importing '{name}' is not allowed.")
        return __import__(name, custom_globals, custom_locals, fromlist or (), level)

    @staticmethod
    def safe_builtins() -> Dict[str, Any]:
        """Creates a dictionary of built-in functions with unsafe ones removed.

        Returns:
            A dictionary of safe built-in functions and objects.
        """
        import builtins

        safe_builtins = {
            k: v
            for k, v in builtins.__dict__.items()
            if k not in SandboxPython.UNSAFE_BUILTINS
        }
        safe_builtins["__import__"] = SandboxPython.restricted_import
        return safe_builtins

    @staticmethod
    def exec(code: str, locals: Dict[str, Any]) -> None:
        """Executes Python code in a restricted environment.

        Args:
            code: The Python code to execute as a string.
            locals: A dictionary that will be used for local variable storage.
        """
        exec(code, {"__builtins__": SandboxPython.safe_builtins()}, locals)


class CodeInterpreterTool(BaseTool):
    """A tool for executing Python code in isolated environments.

    This tool provides functionality to run Python code either in a Docker container
    for safe isolation or directly in a restricted sandbox. It can handle installing
    Python packages and executing arbitrary Python code.
    """

    name: str = "Code Interpreter"
    description: str = "Interprets Python3 code strings with a final print statement."
    args_schema: Type[BaseModel] = CodeInterpreterSchema
    default_image_tag: str = "code-interpreter:latest"
    code: Optional[str] = None
    user_dockerfile_path: Optional[str] = None
    user_docker_base_url: Optional[str] = None
    unsafe_mode: bool = False

    @staticmethod
    def _get_installed_package_path() -> str:
        """Gets the installation path of the crewai_tools package.

        Returns:
            The directory path where the package is installed.
        """
        spec = importlib.util.find_spec("crewai_tools")
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
                    )

            client.images.build(
                path=dockerfile_path,
                tag=self.default_image_tag,
                rm=True,
            )

    def _run(self, **kwargs) -> str:
        """Runs the code interpreter tool with the provided arguments.

        Args:
            **kwargs: Keyword arguments that should include 'code' and 'libraries_used'.

        Returns:
            The output of the executed code as a string.
        """
        code = kwargs.get("code", self.code)
        libraries_used = kwargs.get("libraries_used", [])

        if self.unsafe_mode:
            return self.run_code_unsafe(code, libraries_used)
        else:
            return self.run_code_safety(code, libraries_used)

    def _install_libraries(self, container: Container, libraries: List[str]) -> None:
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

    def _check_docker_available(self) -> bool:
        """Checks if Docker is available and running on the system.

        Attempts to run the 'docker info' command to verify Docker availability.
        Prints appropriate messages if Docker is not installed or not running.

        Returns:
            True if Docker is available and running, False otherwise.
        """
        import subprocess

        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            Printer.print(
                "Docker is installed but not running or inaccessible.",
                color="bold_purple",
            )
            return False
        except FileNotFoundError:
            Printer.print("Docker is not installed", color="bold_purple")
            return False

    def run_code_safety(self, code: str, libraries_used: List[str]) -> str:
        """Runs code in the safest available environment.

        Attempts to run code in Docker if available, falls back to a restricted
        sandbox if Docker is not available.

        Args:
            code: The Python code to execute as a string.
            libraries_used: A list of Python library names to install before execution.

        Returns:
            The output of the executed code as a string.
        """
        if self._check_docker_available():
            return self.run_code_in_docker(code, libraries_used)
        else:
            return self.run_code_in_restricted_sandbox(code)

    def run_code_in_docker(self, code: str, libraries_used: List[str]) -> str:
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

    def run_code_in_restricted_sandbox(self, code: str) -> str:
        """Runs Python code in a restricted sandbox environment.

        Executes the code with restricted access to potentially dangerous modules and
        built-in functions for basic safety when Docker is not available.

        Args:
            code: The Python code to execute as a string.

        Returns:
            The value of the 'result' variable from the executed code,
            or an error message if execution failed.
        """
        Printer.print("Running code in restricted sandbox", color="yellow")
        exec_locals = {}
        try:
            SandboxPython.exec(code=code, locals=exec_locals)
            return exec_locals.get("result", "No result variable found.")
        except Exception as e:
            return f"An error occurred: {str(e)}"

    def run_code_unsafe(self, code: str, libraries_used: List[str]) -> str:
        """Runs code directly on the host machine without any safety restrictions.

        WARNING: This mode is unsafe and should only be used in trusted environments
        with code from trusted sources.

        Args:
            code: The Python code to execute as a string.
            libraries_used: A list of Python library names to install before execution.

        Returns:
            The value of the 'result' variable from the executed code,
            or an error message if execution failed.
        """

        Printer.print("WARNING: Running code in unsafe mode", color="bold_magenta")
        # Install libraries on the host machine
        for library in libraries_used:
            os.system(f"pip install {library}")

        # Execute the code
        try:
            exec_locals = {}
            exec(code, {}, exec_locals)
            return exec_locals.get("result", "No result variable found.")
        except Exception as e:
            return f"An error occurred: {str(e)}"
