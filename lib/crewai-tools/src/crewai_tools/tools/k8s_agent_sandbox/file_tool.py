import base64
import shlex
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
            safe_path = shlex.quote(file_path)

            if action == "read":
                inner_cmd = f"cat {safe_path}"

            elif action in ["write", "append"]:
                if not content:
                    return "Error: 'content' parameter is required for the 'write' action."
                # Base64 encode the content to safely write multiline text/symbols
                encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')
                redirect_operator = ">" if action == "write" else ">>"
                inner_cmd = f"echo '{encoded_content}' | base64 -d {redirect_operator} {safe_path}"

            elif action == "delete":
                inner_cmd = f"rm -rf {safe_path}"

            elif action == "list":
                inner_cmd = f"ls -la {file_path}"

            else:
                return (
                    f"Error: Unknown action '{action}'. "
                    "Supported actions are 'read', 'write', 'append', 'delete', and 'list'."
                )
            safe_inner_cmd = shlex.quote(inner_cmd)
            response = sandbox.commands.run(f"sh -c {safe_inner_cmd}")
            if response.exit_code == 0:
                if action in ["read", "list"]:
                    return response.stdout
                return f"Successfully executed '{action}' on {file_path}."
            else:
                return f"Error executing '{action}' on {file_path}:\n{response.stderr}"
        finally:
            self._release_sandbox(sandbox, should_terminate)
