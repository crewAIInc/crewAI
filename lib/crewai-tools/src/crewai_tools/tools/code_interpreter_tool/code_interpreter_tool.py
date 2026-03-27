"""Code Interpreter Tool for executing Python code in isolated environments.

This module provides a tool for executing Python code either in a Docker container,
a sandlock process sandbox, or directly in a restricted sandbox. It includes mechanisms
for blocking potentially unsafe operations and importing restricted modules.

Execution backends (in order of preference):
    1. Docker: Full container isolation (~200ms startup)
    2. Sandlock: Kernel-level process sandbox via Landlock + seccomp-bpf (~1ms startup)
    3. Unsafe: Direct execution on the host (no isolation, trusted code only)

Example usage::

    from crewai_tools import CodeInterpreterTool

    # Auto-select best available backend (Docker > Sandlock > error)
    tool = CodeInterpreterTool()

    # Explicitly use sandlock backend
    tool = CodeInterpreterTool(
        execution_backend="sandlock",
        sandbox_fs_read=["/usr/lib/python3", "/workspace"],
        sandbox_fs_write=["/workspace/output"],
        sandbox_max_memory_mb=512,
    )

    # Use unsafe mode (only for trusted code)
    tool = CodeInterpreterTool(unsafe_mode=True)
"""

import importlib.util
import os
import platform
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any, ClassVar, Literal, TypedDict

from crewai.tools import BaseTool
from docker import (  # type: ignore[import-untyped]
    DockerClient,
    from_env as docker_from_env,
)
from docker.errors import ImageNotFound, NotFound  # type: ignore[import-untyped]
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


class SandboxPython:
    """INSECURE: A restricted Python execution environment with known vulnerabilities.

    WARNING: This class does NOT provide real security isolation and is vulnerable to
    sandbox escape attacks via Python object introspection. Attackers can recover the
    original __import__ function and bypass all restrictions.

    DO NOT USE for untrusted code execution. Use Docker containers or sandlock instead.

    This class attempts to restrict access to dangerous modules and built-in functions
    but provides no real security boundary against a motivated attacker.
    """

    BLOCKED_MODULES: ClassVar[set[str]] = {
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

    UNSAFE_BUILTINS: ClassVar[set[str]] = {
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
        custom_globals: dict[str, Any] | None = None,
        custom_locals: dict[str, Any] | None = None,
        fromlist: list[str] | None = None,
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
    def safe_builtins() -> dict[str, Any]:
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
    def exec(code: str, locals_: dict[str, Any]) -> None:
        """Executes Python code in a restricted environment.

        Args:
            code: The Python code to execute as a string.
            locals_: A dictionary that will be used for local variable storage.
        """
        exec(code, {"__builtins__": SandboxPython.safe_builtins()}, locals_)  # noqa: S102


class CodeInterpreterTool(BaseTool):
    """A tool for executing Python code in isolated environments.

    This tool provides functionality to run Python code either in a Docker container
    for safe isolation, in a sandlock process sandbox for lightweight kernel-level
    isolation, or directly in a restricted sandbox. It can handle installing
    Python packages and executing arbitrary Python code.

    Attributes:
        execution_backend: The execution backend to use. One of ``"auto"``,
            ``"docker"``, ``"sandlock"``, or ``"unsafe"``. Defaults to ``"auto"``
            which tries Docker first, then sandlock, then raises an error.
        sandbox_fs_read: List of filesystem paths to allow read access in sandlock.
        sandbox_fs_write: List of filesystem paths to allow write access in sandlock.
        sandbox_max_memory_mb: Maximum memory in MB for sandlock execution.
        sandbox_max_processes: Maximum number of processes for sandlock execution.
        sandbox_timeout: Timeout in seconds for sandlock execution.

    Example::

        # Auto-select best available backend
        tool = CodeInterpreterTool()
        result = tool.run(code="print('hello')", libraries_used=[])

        # Explicitly use sandlock with custom policy
        tool = CodeInterpreterTool(
            execution_backend="sandlock",
            sandbox_fs_read=["/usr/lib/python3"],
            sandbox_fs_write=["/tmp/output"],
            sandbox_max_memory_mb=256,
        )
        result = tool.run(code="print(2 + 2)", libraries_used=[])
    """

    name: str = "Code Interpreter"
    description: str = "Interprets Python3 code strings with a final print statement."
    args_schema: type[BaseModel] = CodeInterpreterSchema
    default_image_tag: str = "code-interpreter:latest"
    code: str | None = None
    user_dockerfile_path: str | None = None
    user_docker_base_url: str | None = None
    unsafe_mode: bool = False

    execution_backend: Literal["auto", "docker", "sandlock", "unsafe"] = "auto"
    sandbox_fs_read: list[str] = Field(default_factory=list)
    sandbox_fs_write: list[str] = Field(default_factory=list)
    sandbox_max_memory_mb: int | None = None
    sandbox_max_processes: int | None = None
    sandbox_timeout: int | None = None

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

        # Handle legacy unsafe_mode flag
        if self.unsafe_mode or self.execution_backend == "unsafe":
            return self.run_code_unsafe(code, libraries_used)

        if self.execution_backend == "docker":
            return self.run_code_in_docker(code, libraries_used)

        if self.execution_backend == "sandlock":
            return self.run_code_in_sandlock(code, libraries_used)

        # Auto mode: try Docker first, then sandlock, then raise error
        return self.run_code_safety(code, libraries_used)

    @staticmethod
    def _install_libraries(container: Any, libraries: list[str]) -> None:
        """Installs required Python libraries in the Docker container.

        Args:
            container: The Docker container where libraries will be installed.
            libraries: A list of library names to install using pip.
        """
        for library in libraries:
            container.exec_run(["pip", "install", library])

    def _init_docker_container(self) -> Any:
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
            volumes={current_path: {"bind": "/workspace", "mode": "rw"}},
        )

    @staticmethod
    def _check_docker_available() -> bool:
        """Checks if Docker is available and running on the system.

        Attempts to run the 'docker info' command to verify Docker availability.
        Prints appropriate messages if Docker is not installed or not running.

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
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            Printer.print(
                "Docker is installed but not running or inaccessible.",
                color="bold_purple",
            )
            return False
        except FileNotFoundError:
            Printer.print("Docker is not installed", color="bold_purple")
            return False

    @staticmethod
    def _check_sandlock_available() -> bool:
        """Checks if sandlock is installed and the system supports it.

        Verifies that:
        1. The sandlock package is importable
        2. The system is running Linux (sandlock requires Linux kernel features)

        Returns:
            True if sandlock is available and the system supports it, False otherwise.
        """
        if platform.system() != "Linux":
            Printer.print(
                "Sandlock requires Linux (Landlock + seccomp-bpf). "
                "Use Docker on macOS/Windows.",
                color="bold_purple",
            )
            return False

        if importlib.util.find_spec("sandlock") is None:
            Printer.print(
                "Sandlock is not installed. Install with: pip install sandlock",
                color="bold_purple",
            )
            return False

        return True

    def _build_sandlock_policy(self, work_dir: str) -> Any:
        """Builds a sandlock Policy with the configured sandbox parameters.

        Constructs a sandlock Policy object using the tool's configuration for
        filesystem access, memory limits, process limits, and other constraints.

        Args:
            work_dir: The working directory for the sandbox (writable).

        Returns:
            A sandlock Policy object configured with the appropriate restrictions.
        """
        from sandlock import Policy  # type: ignore[import-untyped,import-not-found]

        # Default readable paths for Python execution
        default_readable = [
            "/usr",
            "/lib",
            "/lib64",
            "/etc/alternatives",
        ]

        # Add Python-specific paths
        python_path = os.path.dirname(os.path.dirname(sys.executable))
        if python_path not in default_readable:
            default_readable.append(python_path)

        # Include site-packages for installed libraries
        for path in sys.path:
            if path and os.path.isdir(path) and path not in default_readable:
                default_readable.append(path)

        fs_readable = list(set(default_readable + self.sandbox_fs_read))
        fs_writable = list(set([work_dir, *self.sandbox_fs_write]))

        policy_kwargs: dict[str, Any] = {
            "fs_readable": fs_readable,
            "fs_writable": fs_writable,
            "isolate_ipc": True,
            "clean_env": True,
            "env": {"PATH": "/usr/bin:/bin", "HOME": work_dir},
        }

        if self.sandbox_max_memory_mb is not None:
            policy_kwargs["max_memory"] = f"{self.sandbox_max_memory_mb}M"

        if self.sandbox_max_processes is not None:
            policy_kwargs["max_processes"] = self.sandbox_max_processes

        return Policy(**policy_kwargs)

    def run_code_in_sandlock(self, code: str, libraries_used: list[str]) -> str:
        """Runs Python code in a sandlock process sandbox.

        Uses sandlock's Landlock + seccomp-bpf kernel-level isolation to execute
        code in a confined process. This provides stronger isolation than the
        Python-level SandboxPython (which is vulnerable to escape attacks) while
        being much lighter than Docker (~1ms vs ~200ms startup).

        Libraries are installed in a temporary directory before sandbox activation.

        Args:
            code: The Python code to execute as a string.
            libraries_used: A list of Python library names to install before execution.

        Returns:
            The output of the executed code as a string, or an error message
            if execution failed.

        Raises:
            RuntimeError: If sandlock is not available or the system doesn't support it.
        """
        if not self._check_sandlock_available():
            raise RuntimeError(
                "Sandlock is not available. Ensure sandlock is installed "
                "(pip install sandlock) and you are running on Linux 5.13+."
            )

        from sandlock import Sandbox  # type: ignore[import-untyped,import-not-found]

        Printer.print(
            "Running code in sandlock sandbox (Landlock + seccomp-bpf)",
            color="bold_blue",
        )

        with tempfile.TemporaryDirectory(prefix="crewai_sandbox_") as work_dir:
            # Install libraries before entering the sandbox
            if libraries_used:
                Printer.print(
                    f"Installing libraries: {', '.join(libraries_used)}",
                    color="bold_purple",
                )
                for library in libraries_used:
                    subprocess.run(  # noqa: S603
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--target",
                            os.path.join(work_dir, "libs"),
                            library,
                        ],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

            # Write the code to a temporary file
            code_file = os.path.join(work_dir, "script.py")
            with open(code_file, "w") as f:  # noqa: PTH123
                f.write(code)

            # Build the sandbox policy
            policy = self._build_sandlock_policy(work_dir)

            # Build the command with PYTHONPATH for installed libraries
            env_pythonpath = os.path.join(work_dir, "libs")
            cmd = [
                sys.executable,
                "-c",
                (
                    f"import sys; sys.path.insert(0, '{env_pythonpath}'); "
                    f"exec(open('{code_file}').read())"
                ),
            ]

            timeout = self.sandbox_timeout if self.sandbox_timeout is not None else 60

            try:
                result = Sandbox(policy).run(cmd, timeout=timeout)
                output = result.stdout if hasattr(result, "stdout") else str(result)
                if hasattr(result, "returncode") and result.returncode != 0:
                    stderr = result.stderr if hasattr(result, "stderr") else ""
                    return (
                        f"Something went wrong while running the code: "
                        f"\n{stderr or output}"
                    )
                return output
            except Exception as e:
                return f"An error occurred in sandlock sandbox: {e!s}"

    def run_code_safety(self, code: str, libraries_used: list[str]) -> str:
        """Runs code in the safest available environment.

        Tries execution backends in order of isolation strength:
        1. Docker (full container isolation)
        2. Sandlock (kernel-level process sandbox, Linux only)

        Fails closed if neither backend is available.

        Args:
            code: The Python code to execute as a string.
            libraries_used: A list of Python library names to install before execution.

        Returns:
            The output of the executed code as a string.

        Raises:
            RuntimeError: If no secure execution backend is available.
        """
        if self._check_docker_available():
            return self.run_code_in_docker(code, libraries_used)

        if self._check_sandlock_available():
            Printer.print(
                "Docker unavailable, falling back to sandlock sandbox.",
                color="bold_yellow",
            )
            return self.run_code_in_sandlock(code, libraries_used)

        error_msg = (
            "No secure execution backend is available. "
            "Install Docker (https://docs.docker.com/get-docker/) for full container isolation, "
            "or install sandlock (pip install sandlock) on Linux 5.13+ for lightweight "
            "kernel-level sandboxing via Landlock + seccomp-bpf. "
            "Alternatively, use unsafe_mode=True or execution_backend='unsafe' "
            "if you trust the code source and understand the security risks."
        )
        Printer.print(error_msg, color="bold_red")
        raise RuntimeError(error_msg)

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

        exec_result: Any = container.exec_run(["python3", "-c", code])

        container.stop()
        container.remove()

        if exec_result.exit_code != 0:
            return f"Something went wrong while running the code: \n{exec_result.output.decode('utf-8')}"
        return str(exec_result.output.decode("utf-8"))

    @staticmethod
    def run_code_in_restricted_sandbox(code: str) -> str:
        """DEPRECATED AND INSECURE: Runs Python code in a restricted sandbox environment.

        WARNING: This method is vulnerable to sandbox escape attacks via Python object
        introspection and should NOT be used for untrusted code execution. It has been
        deprecated and is only kept for backward compatibility with trusted code.

        The "restricted" environment can be bypassed by attackers who can:
        - Use object graph introspection to recover the original __import__ function
        - Access any Python module including os, subprocess, sys, etc.
        - Execute arbitrary commands on the host system

        Use run_code_in_docker() or run_code_in_sandlock() for secure code execution,
        or run_code_unsafe() if you explicitly acknowledge the security risks.

        Args:
            code: The Python code to execute as a string.

        Returns:
            The value of the 'result' variable from the executed code,
            or an error message if execution failed.
        """
        Printer.print(
            "WARNING: Running code in INSECURE restricted sandbox (vulnerable to escape attacks)",
            color="bold_red",
        )
        exec_locals: dict[str, Any] = {}
        try:
            SandboxPython.exec(code=code, locals_=exec_locals)
            return exec_locals.get("result", "No result variable found.")  # type: ignore[no-any-return]
        except Exception as e:
            return f"An error occurred: {e!s}"

    @staticmethod
    def run_code_unsafe(code: str, libraries_used: list[str]) -> str:
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
            subprocess.run(  # noqa: S603
                [sys.executable, "-m", "pip", "install", library], check=False
            )

        # Execute the code
        try:
            exec_locals: dict[str, Any] = {}
            exec(code, {}, exec_locals)  # noqa: S102
            return exec_locals.get("result", "No result variable found.")  # type: ignore[no-any-return]
        except Exception as e:
            return f"An error occurred: {e!s}"
