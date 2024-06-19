from typing import Optional, Type

import docker
from crewai_tools.tools.base_tool import BaseTool
from pydantic.v1 import BaseModel, Field


class FixedCodeInterpreterSchemaSchema(BaseModel):
    """Input for DirectoryReadTool."""

    pass


class CodeInterpreterSchema(FixedCodeInterpreterSchemaSchema):
    """Input for DirectoryReadTool."""

    code: str = Field(
        ...,
        description="Python3 code used to be interpreted in the Docker container and output the result",
    )


class CodeInterpreterTool(BaseTool):
    name: str = "Code Interpreter"
    description: str = "Interprets Python code in a Docker container"
    args_schema: Type[BaseModel] = CodeInterpreterSchema
    code: Optional[str] = None

    def __init__(self, code: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if code is not None:
            self.code = code
            self.description = (
                "A tool that can be used to run Python code in a Docker container"
            )
            self.args_schema = FixedCodeInterpreterSchemaSchema
            self._generate_description()

    def _run(self, **kwargs):
        code = kwargs.get("code", self.code)
        return self.run_code_in_docker(code)

    def run_code_in_docker(self, code):
        client = docker.from_env()
        container = client.containers.run(
            "code-interpreter",
            command=f'python3 -c "{code}"',
            detach=True,
            working_dir="/workspace",
        )

        result = container.logs().decode("utf-8")

        container.stop()
        container.remove()

        return result
