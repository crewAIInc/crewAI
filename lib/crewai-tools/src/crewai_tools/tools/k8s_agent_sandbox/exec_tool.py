from typing import Any

from pydantic import BaseModel, Field
from k8s_agent_sandbox.sandbox import Sandbox

from crewai_tools.tools.k8s_agent_sandbox.base_tool import (
    K8sAgentSandboxBaseTool,
    DEFAULT_TOOL_TIMEOUT_SEC,
)


class k8sAgentSandboxExecToolSchema(BaseModel):
    command: str = Field(..., description="Shell command to execute in the sandbox.")
    timeout: int = Field(
        default=DEFAULT_TOOL_TIMEOUT_SEC,
        description="Maximum seconds to wait for the command to finish.",
    )


class K8sAgentSandboxExecTool(K8sAgentSandboxBaseTool):
    name: str = "K8s Agent Sandbox Exec Tool"
    description: str = "Executes shell commands inside an isolated Kubernetes pod sandbox."
    args_schema: type[BaseModel] = k8sAgentSandboxExecToolSchema

    def _run_with_sandbox(self, sandbox: Sandbox, *args, **kwargs) -> dict[str, Any]:
      return self._run_command(sandbox, *args, **kwargs)

    def _run_command(
        self,
        sandbox: Sandbox,
        command: str,
        timeout: int,
    ) -> dict[str, Any]:
        result = sandbox.commands.run(command, timeout=timeout)
        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }


