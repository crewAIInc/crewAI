import importlib.util
import os
from typing import List, Optional, Type

import docker
from crewai_tools.tools.base_tool import BaseTool
from pydantic.v1 import BaseModel, Field


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
    code: Optional[str] = None

    @staticmethod
    def _get_installed_package_path():
        spec = importlib.util.find_spec("crewai_tools")
        return os.path.dirname(spec.origin)

    def _verify_docker_image(self) -> None:
        """
        Verify if the Docker image is available
        """
        image_tag = "code-interpreter:latest"
        client = docker.from_env()

        try:
            client.images.get(image_tag)

        except docker.errors.ImageNotFound:
            package_path = self._get_installed_package_path()
            dockerfile_path = os.path.join(package_path, "tools/code_interpreter_tool")
            if not os.path.exists(dockerfile_path):
                raise FileNotFoundError(f"Dockerfile not found in {dockerfile_path}")

            client.images.build(
                path=dockerfile_path,
                tag=image_tag,
                rm=True,
            )

    def _run(self, **kwargs) -> str:
        code = kwargs.get("code", self.code)
        libraries_used = kwargs.get("libraries_used", [])
        return self.run_code_in_docker(code, libraries_used)

    def _install_libraries(
        self, container: docker.models.containers.Container, libraries: List[str]
    ) -> None:
        """
        Install missing libraries in the Docker container
        """
        for library in libraries:
            container.exec_run(f"pip install {library}")

    def _init_docker_container(self) -> docker.models.containers.Container:
        client = docker.from_env()
        return client.containers.run(
            "code-interpreter",
            detach=True,
            tty=True,
            working_dir="/workspace",
            name="code-interpreter",
        )

    def run_code_in_docker(self, code: str, libraries_used: List[str]) -> str:
        self._verify_docker_image()
        container = self._init_docker_container()
        self._install_libraries(container, libraries_used)

        cmd_to_run = f'python3 -c "{code}"'
        exec_result = container.exec_run(cmd_to_run)

        container.stop()
        container.remove()

        if exec_result.exit_code != 0:
            return f"Something went wrong while running the code: \n{exec_result.output.decode('utf-8')}"
        return exec_result.output.decode("utf-8")
