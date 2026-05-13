import base64
from typing import Any
from crewai.tools import BaseTool
from pydantic import Field


class K8sSandboxPythonTool(BaseTool):
    name: str = "K8sSandboxPythonTool"
    description: str = (
        "Executes Python code inside an isolated Kubernetes pod sandbox. "
        "Input should be a string containing raw Python code."
    )

    namespace: str = Field(default="default", description="K8s namespace")

    def _run(self, code: str, **kwargs: Any) -> str:
        try:
            import k8s_agent_sandbox
        except ImportError:
            raise ImportError(
                "The k8s_agent_sandbox SDK is not installed. "
                "Please install it using: uv add 'crewai-tools[k8s_agent_sandbox]'"
            )

        client = k8s_agent_sandbox.SandboxClient()
        sandbox = client.create_sandbox("simple-sandbox-template")

        # 1. Base64 encode the raw Python code safely
        encoded_code = base64.b64encode(code.encode('utf-8')).decode('utf-8')

        # 2. Construct a single-line python -c command that decodes and executes the payload.
        # This prevents any single quote (') or double quote (") collisions in the shell.
        safe_command = f'python -c "import base64; exec(base64.b64decode(\'{encoded_code}\').decode(\'utf-8\'))"'

        # 3. Execute the Python script
        exec_response = sandbox.commands.run(safe_command)

        if exec_response.exit_code == 0:
            return exec_response.stdout
        else:
            return f"Python execution failed (Exit Code {exec_response.exit_code}):\n{exec_response.stderr}"
