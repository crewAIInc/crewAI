from typing import TYPE_CHECKING
import time
from pydantic import (
    BaseModel,
    Field,
)

if TYPE_CHECKING:
    from k8s_agent_sandbox.sandbox import Sandbox

from crewai_tools.tools.k8s_agent_sandbox.base_tool import (
    K8sAgentSandboxBaseTool,
    DEFAULT_TOOL_TIMEOUT_SEC,
    create_timeout_tracker,
)


class k8sAgentSandboxPythonToolSchema(BaseModel):
    code: str = Field(..., description="Shell command to execute in the sandbox.")
    timeout: int = Field(
        default=DEFAULT_TOOL_TIMEOUT_SEC,
        description="Maximum seconds to wait for the command to finish.",
    )


class K8sAgentSandboxPythonToolOutput(BaseModel):
    exit_code: int | None = Field(default=None, description="The exit code of the Python script execution.")
    stdout: str | None = Field(default=None, description="The standard output produced by the Python script.")
    stderr: str | None = Field(default=None, description="The standard error output produced by the Python script.")


class K8sAgentSandboxPythonTool(K8sAgentSandboxBaseTool):
    name: str = "K8s Agent Sandbox Python Tool"
    description: str = (
        "Executes Python code inside an isolated K8s Agent Sandbox. "
        "Input should be a string containing raw Python code."
    )
    args_schema: type[BaseModel] = k8sAgentSandboxPythonToolSchema

    def _run_with_sandbox(
        self, sandbox: "Sandbox", code: str, timeout: int
    ) -> K8sAgentSandboxPythonToolOutput:

        timeout_tracker = create_timeout_tracker(timeout)

        tmp_file_path = f"/tmp/crewai-{int(time.time())}.py"

        sandbox.files.write(tmp_file_path, code.encode("utf-8"), timeout=timeout_tracker())

        result = sandbox.commands.run(f"python3 {tmp_file_path}", timeout=timeout_tracker())

        return K8sAgentSandboxPythonToolOutput(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )
