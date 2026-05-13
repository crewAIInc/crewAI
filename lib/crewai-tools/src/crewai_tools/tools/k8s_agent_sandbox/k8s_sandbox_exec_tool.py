from typing import Any, Optional
from crewai.tools import BaseTool
from pydantic import Field


class K8sSandboxExecTool(BaseTool):
    name: str = "K8sSandboxExecTool"
    description: str = "Executes shell commands inside an isolated Kubernetes pod sandbox."

    # Define your configuration fields (e.g., namespace, pod execution settings)
    namespace: str = Field(default="default", description="K8s namespace")

    def _run(self, command: str, **kwargs: Any) -> str:
        try:
            import k8s_agent_sandbox
        except ImportError:
            raise ImportError(
                "The k8s_agent_sandbox SDK is not installed. "
                "Please install it using: uv add 'crewai-tools[k8s_agent_sandbox]'"
            )

        client = k8s_agent_sandbox.SandboxClient()
        sandbox = client.create_sandbox("simple-sandbox-template")
        response = sandbox.commands.run(command)
        if response.exit_code == 0:
            return response.stdout
        else:
            return f"Python execution failed (Exit Code {response.exit_code}):\n{response.stderr}"
