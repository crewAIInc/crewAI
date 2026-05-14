import base64
from typing import Any, Optional
from crewai.tools import BaseTool
from pydantic import Field
try:
    import k8s_agent_sandbox
except ImportError:
    raise ImportError(
        "The k8s_agent_sandbox SDK is not installed. "
        "Please install it using: uv add 'crewai-tools[k8s_agent_sandbox]'"
    )


class K8sSandboxFileTool(BaseTool):
    name: str = "K8sSandboxFileTool"
    description: str = (
        "Reads, writes, or lists files inside an isolated Kubernetes pod sandbox. "
        "Requires an 'action' ('read', 'write', or 'list') and a 'file_path'."
    )

    namespace: str = Field(default="default", description="K8s namespace")
    client: k8s_agent_sandbox.SandboxClient = k8s_agent_sandbox.SandboxClient()
    sandbox: k8s_agent_sandbox.sandbox_client.Sandbox | None = client.create_sandbox("simple-sandbox-template")


    def _run(self, action: str, file_path: str, content: Optional[str] = None, **kwargs: Any) -> str:
        action = action.lower()

        if action == "read":
            response = self.sandbox.commands.run(f'sh -c "cat {file_path}"')
            if response.exit_code == 0:
                return response.stdout
            return f"Error reading file {file_path}:\n{response.stderr}"

        elif action == "write":
            if not content:
                return "Error: 'content' parameter is required for the 'write' action."

            # Base64 encode the content to safely write multiline text/symbols
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
            response = self.sandbox.commands.run(f"sh -c \"echo '{encoded_content}' | base64 -d > {file_path}\"")

            if response.exit_code == 0:
                return f"Successfully wrote content to {file_path}."
            return f"Error writing to file {file_path}:\n{response.stderr}"

        elif action == "list":
            # Also wrapping list in sh -c just to be safe
            response = self.sandbox.commands.run(f'sh -c "ls -la {file_path}"')
            if response.exit_code == 0:
                return response.stdout
            return f"Error listing path {file_path}:\n{response.stderr}"

        else:
            return f"Error: Unknown action '{action}'. Supported actions are 'read', 'write', and 'list'."
