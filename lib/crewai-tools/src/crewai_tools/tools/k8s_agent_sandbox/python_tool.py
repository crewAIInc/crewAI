import base64
from typing import Any

from crewai_tools.tools.k8s_agent_sandbox.base_tool import K8sBaseTool


class K8sPythonTool(K8sBaseTool):
    name: str = "K8sPythonTool"
    description: str = (
        "Executes Python code inside an isolated Kubernetes pod sandbox. "
        "Input should be a string containing raw Python code."
    )

    def _run(self, code: str, **kwargs: Any) -> str:
        sandbox, should_terminate = self._get_sandbox()
        try:
            # Base64 encode the raw Python code safely
            encoded_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')

            # Construct a single-line python -c command that decodes and executes the payload.
            # This prevents any single quote (') or double quote (") collisions in the shell.
            safe_command = f'python -c "import base64; exec(base64.b64decode(\'{encoded_code}\').decode(\'utf-8\'))"'
            exec_response = sandbox.commands.run(safe_command)

            sandbox.terminate()
            if exec_response.exit_code == 0:
                return exec_response.stdout
            else:
                return f"Python execution failed (Exit Code {exec_response.exit_code}):\n{exec_response.stderr}"
        finally:
            self._release_sandbox(sandbox, should_terminate)
