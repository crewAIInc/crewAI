from typing import Any

from crewai_tools.tools.k8s_agent_sandbox.base_tool import K8sBaseTool


class K8sExecTool(K8sBaseTool):
    name: str = "K8sExecTool"
    description: str = "Executes shell commands inside an isolated Kubernetes pod sandbox."

    def _run(self, command: str, **kwargs: Any) -> str:
        sandbox, should_terminate = self._get_sandbox()
        try:
            response = sandbox.commands.run(command)
            if response.exit_code == 0:
                return response.stdout
            else:
                return f"Command execution failed (Exit Code {response.exit_code}):\n{response.stderr}"
        finally:
            self._release_sandbox(sandbox, should_terminate)
