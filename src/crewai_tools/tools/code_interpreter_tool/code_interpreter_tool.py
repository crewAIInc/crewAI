import importlib.util
import os
from typing import List, Optional, Type

from crewai.tools import BaseTool
from docker import from_env as docker_from_env
from docker import DockerClient
from docker.models.containers import Container
from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container
from pydantic import BaseModel, Field


class CodeInterpreterSchema(BaseModel):
    """Input for CodeInterpreterTool."""

    code: str = Field(
        ...,
        description="Python3 code used to be interpreted in the Docker container. ALWAYS PRINT the final result and the output of the code",
    )

    libraries_used: List[str] = Field(
        ...,
        description="List of libraries used in the code with proper installing names separated by commas. Example: numpy,pandas,beautifulsoup4",
    )


class CodeInterpreterTool(BaseTool):
    name: str = "Code Interpreter"
    description: str = "Interprets Python3 code strings with a final print statement."
    args_schema: Type[BaseModel] = CodeInterpreterSchema
    default_image_tag: str = "code-interpreter:latest"
    code: Optional[str] = None
    user_dockerfile_path: Optional[str] = None
    user_docker_base_url: Optional[str] = None
    unsafe_mode: bool = False

    @staticmethod
    def _get_installed_package_path():
        spec = importlib.util.find_spec("crewai_tools")
        return os.path.dirname(spec.origin)

    def _verify_docker_image(self) -> None:
        """
        Verify if the Docker image is available. Optionally use a user-provided Dockerfile.
        """

        client = (
            docker_from_env()
            if self.user_docker_base_url == None
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
        code = kwargs.get("code", self.code)
        libraries_used = kwargs.get("libraries_used", [])

        if self.unsafe_mode:
            return self.run_code_unsafe(code, libraries_used)
        else:
            return self.run_code_in_docker(code, libraries_used)

    def _install_libraries(self, container: Container, libraries: List[str]) -> None:
        """
        Install missing libraries in the Docker container
        """
        for library in libraries:
            container.exec_run(["pip", "install", library])

    def _init_docker_container(self) -> Container:
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

    def run_code_in_docker(self, code: str, libraries_used: List[str]) -> str:
        self._verify_docker_image()
        container = self._init_docker_container()
        self._install_libraries(container, libraries_used)

        exec_result = container.exec_run(["python3", "-c", code])

        container.stop()
        container.remove()

        if exec_result.exit_code != 0:
            return f"Something went wrong while running the code: \n{exec_result.output.decode('utf-8')}"
        return exec_result.output.decode("utf-8")

    def run_code_unsafe(self, code: str, libraries_used: List[str]) -> str:
        """
        Run the code directly on the host machine (unsafe mode).
        """
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