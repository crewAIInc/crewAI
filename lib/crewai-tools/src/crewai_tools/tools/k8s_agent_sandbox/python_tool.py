from typing import Any

from pydantic import (
    BaseModel,
    Field,
)
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


class K8sAgentSandboxPythonTool(K8sAgentSandboxBaseTool):
    name: str = "K8s Agent Sandbox Python Tool"
    description: str = (
        "Executes Python code inside an isolated K8s Agent Sandbox. "
        "Input should be a string containing raw Python code."
    )
    args_schema: type[BaseModel] = k8sAgentSandboxPythonToolSchema

    def _run_with_sandbox(
        self, sandbox: Sandbox, code: str, timeout: int
    ) -> dict[str, Any]:

        timeout_tracker = create_timeout_tracker(timeout)
        sandbox.files.write("main.py", code, timeout=timeout_tracker())
        result = sandbox.commands.run("python3 main.py", timeout=timeout_tracker())

        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
