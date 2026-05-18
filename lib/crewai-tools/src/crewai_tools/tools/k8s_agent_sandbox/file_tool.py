import base64
from typing import Any, Optional
from pydantic import Field
from crewai_tools.tools.k8s_agent_sandbox.base_tool import K8sBaseTool


class K8sFileTool(K8sBaseTool):
    name: str = "K8sFileTool"
    description: str = (
        "Reads, writes, or lists files inside an isolated Kubernetes pod sandbox. "
        "Requires an 'action' ('read', 'write', or 'list') and a 'file_path'."
    )
    persistent: bool = Field(
        default=True,
        description=(
            "File tool must use a persistent sandbox."
        ),
    )

    def _run(self, action: str, file_path: str, content: Optional[str] = None, **kwargs: Any) -> str:
        sandbox, should_terminate = self._get_sandbox()

        try:
            action = action.lower()
            if action == "read":
                response = sandbox.commands.run(f'sh -c "cat {file_path}"')
                if response.exit_code == 0:
                    return response.stdout
                return f"Error reading file {file_path}:\n{response.stderr}"

            elif action == "write":
                if not content:
                    return "Error: 'content' parameter is required for the 'write' action."

                # Base64 encode the content to safely write multiline text/symbols
                encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
                response = sandbox.commands.run(f"sh -c \"echo '{encoded_content}' | base64 -d > {file_path}\"")

                if response.exit_code == 0:
                    return f"Successfully wrote content to {file_path}."
                return f"Error writing to file {file_path}:\n{response.stderr}"

            elif action == "list":
                # Also wrapping list in sh -c just to be safe
                response = sandbox.commands.run(f'sh -c "ls -la {file_path}"')
                if response.exit_code == 0:
                    return response.stdout
                return f"Error listing path {file_path}:\n{response.stderr}"

            else:
                return f"Error: Unknown action '{action}'. Supported actions are 'read', 'write', and 'list'."
        finally:
            self._release_sandbox(sandbox, should_terminate)
