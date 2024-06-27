import os
from typing import Optional, Type

import docker
from crewai_tools.tools.base_tool import BaseTool
from pydantic.v1 import BaseModel, Field


class FixedCodeInterpreterSchemaSchema(BaseModel):
    """Input for CodeInterpreterTool."""

    pass


class CodeInterpreterSchema(FixedCodeInterpreterSchemaSchema):
    """Input for CodeInterpreterTool."""

    code: str = Field(
        ...,
        description="Python3 code used to be interpreted in the Docker container. ALWAYS PRINT the final result and the output of the code",
    )
    libraries_used: Optional[str] = Field(
        None,
        description="List of libraries used in the code with proper installing names separated by commas. Example: numpy,pandas,beautifulsoup4",
    )


class CodeInterpreterTool(BaseTool):
    name: str = "Code Interpreter"
    description: str = "Interprets Python code in a Docker container. ALWAYS PRINT the final result and the output of the code"
    args_schema: Type[BaseModel] = CodeInterpreterSchema
    code: Optional[str] = None

    def _verify_docker_image(self) -> None:
        """
        Verify if the Docker image is available
        """
        image_tag = "code-interpreter:latest"

        client = docker.from_env()
        images = client.images.list()
        all_tags = [tag for image in images for tag in image.tags]

        if image_tag not in all_tags:
            client.images.build(
                path=os.path.dirname(os.path.abspath(__file__)),
                tag=image_tag,
                rm=True,
            )

    def __init__(self, code: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if code is not None:
            self._verify_docker_image()
            self.code = code
            self.description = "Interprets Python code in a Docker container. ALWAYS PRINT the final result and the output of the code"
            self.args_schema = FixedCodeInterpreterSchemaSchema
            self._generate_description()

    def _run(self, **kwargs) -> str:
        code = kwargs.get("code", self.code)
        libraries_used = kwargs.get("libraries_used", None)
        return self.run_code_in_docker(code, libraries_used)

    def _install_libraries(
        self, container: docker.models.containers.Container, libraries: list[str]
    ) -> None:
        """
        Install missing libraries in the Docker container
        """
        for library in libraries:
            container.exec_run(f"pip install {library}")

    def _init_docker_container(self) -> docker.models.containers.Container:
        client = docker.from_env()
        return client.containers.run(
            "code-interpreter", detach=True, tty=True, working_dir="/workspace"
        )

    def run_code_in_docker(self, code: str, libraries_used: str) -> str:
        container = self._init_docker_container()

        if libraries_used:
            self._install_libraries(container, libraries_used.split(","))

        cmd_to_run = f'python3 -c "{code}"'
        exec_result = container.exec_run(cmd_to_run)

        container.stop()
        container.remove()

        if exec_result.exit_code != 0:
            return f"Something went wrong while running the code: \n{exec_result.output.decode('utf-8')}"
        return exec_result.output.decode("utf-8")
