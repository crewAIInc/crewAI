"""Toolkit for working with AWS Bedrock Code Interpreter."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

logger = logging.getLogger(__name__)


def extract_output_from_stream(response):
    """Extract output from code interpreter response stream.

    Args:
        response: Response from code interpreter execution

    Returns:
        Extracted output as string
    """
    output = []
    for event in response["stream"]:
        if "result" in event:
            result = event["result"]
            for content_item in result["content"]:
                if content_item["type"] == "text":
                    output.append(content_item["text"])
                if content_item["type"] == "resource":
                    resource = content_item["resource"]
                    if "text" in resource:
                        file_path = resource["uri"].replace("file://", "")
                        file_content = resource["text"]
                        output.append(f"==== File: {file_path} ====\n{file_content}\n")
                    else:
                        output.append(json.dumps(resource))

    return "\n".join(output)


# Input schemas
class ExecuteCodeInput(BaseModel):
    """Input for ExecuteCode."""

    code: str = Field(description="The code to execute")
    language: str = Field(
        default="python", description="The programming language of the code"
    )
    clear_context: bool = Field(
        default=False, description="Whether to clear execution context"
    )
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class ExecuteCommandInput(BaseModel):
    """Input for ExecuteCommand."""

    command: str = Field(description="The command to execute")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class ReadFilesInput(BaseModel):
    """Input for ReadFiles."""

    paths: list[str] = Field(description="List of file paths to read")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class ListFilesInput(BaseModel):
    """Input for ListFiles."""

    directory_path: str = Field(default="", description="Path to the directory to list")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class DeleteFilesInput(BaseModel):
    """Input for DeleteFiles."""

    paths: list[str] = Field(description="List of file paths to delete")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class WriteFilesInput(BaseModel):
    """Input for WriteFiles."""

    files: list[dict[str, str]] = Field(
        description="List of dictionaries with path and text fields"
    )
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class StartCommandInput(BaseModel):
    """Input for StartCommand."""

    command: str = Field(description="The command to execute asynchronously")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class GetTaskInput(BaseModel):
    """Input for GetTask."""

    task_id: str = Field(description="The ID of the task to check")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class StopTaskInput(BaseModel):
    """Input for StopTask."""

    task_id: str = Field(description="The ID of the task to stop")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


# Tool classes
class ExecuteCodeTool(BaseTool):
    """Tool for executing code in various languages."""

    name: str = "execute_code"
    description: str = "Execute code in various languages (primarily Python)"
    args_schema: type[BaseModel] = ExecuteCodeInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(
        self,
        code: str,
        language: str = "python",
        clear_context: bool = False,
        thread_id: str = "default",
    ) -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # Execute code
            response = code_interpreter.invoke(
                method="executeCode",
                params={
                    "code": code,
                    "language": language,
                    "clearContext": clear_context,
                },
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error executing code: {e!s}"

    async def _arun(
        self,
        code: str,
        language: str = "python",
        clear_context: bool = False,
        thread_id: str = "default",
    ) -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(
            code=code,
            language=language,
            clear_context=clear_context,
            thread_id=thread_id,
        )


class ExecuteCommandTool(BaseTool):
    """Tool for running shell commands in the code interpreter environment."""

    name: str = "execute_command"
    description: str = "Run shell commands in the code interpreter environment"
    args_schema: type[BaseModel] = ExecuteCommandInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, command: str, thread_id: str = "default") -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # Execute command
            response = code_interpreter.invoke(
                method="executeCommand", params={"command": command}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error executing command: {e!s}"

    async def _arun(self, command: str, thread_id: str = "default") -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(command=command, thread_id=thread_id)


class ReadFilesTool(BaseTool):
    """Tool for reading content of files in the environment."""

    name: str = "read_files"
    description: str = "Read content of files in the environment"
    args_schema: type[BaseModel] = ReadFilesInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, paths: list[str], thread_id: str = "default") -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # Read files
            response = code_interpreter.invoke(
                method="readFiles", params={"paths": paths}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error reading files: {e!s}"

    async def _arun(self, paths: list[str], thread_id: str = "default") -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(paths=paths, thread_id=thread_id)


class ListFilesTool(BaseTool):
    """Tool for listing files in directories in the environment."""

    name: str = "list_files"
    description: str = "List files in directories in the environment"
    args_schema: type[BaseModel] = ListFilesInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, directory_path: str = "", thread_id: str = "default") -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # List files
            response = code_interpreter.invoke(
                method="listFiles", params={"directoryPath": directory_path}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error listing files: {e!s}"

    async def _arun(self, directory_path: str = "", thread_id: str = "default") -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(directory_path=directory_path, thread_id=thread_id)


class DeleteFilesTool(BaseTool):
    """Tool for removing files from the environment."""

    name: str = "delete_files"
    description: str = "Remove files from the environment"
    args_schema: type[BaseModel] = DeleteFilesInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, paths: list[str], thread_id: str = "default") -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # Remove files
            response = code_interpreter.invoke(
                method="removeFiles", params={"paths": paths}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error deleting files: {e!s}"

    async def _arun(self, paths: list[str], thread_id: str = "default") -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(paths=paths, thread_id=thread_id)


class WriteFilesTool(BaseTool):
    """Tool for creating or updating files in the environment."""

    name: str = "write_files"
    description: str = "Create or update files in the environment"
    args_schema: type[BaseModel] = WriteFilesInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, files: list[dict[str, str]], thread_id: str = "default") -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # Write files
            response = code_interpreter.invoke(
                method="writeFiles", params={"content": files}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error writing files: {e!s}"

    async def _arun(
        self, files: list[dict[str, str]], thread_id: str = "default"
    ) -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(files=files, thread_id=thread_id)


class StartCommandTool(BaseTool):
    """Tool for starting long-running commands asynchronously."""

    name: str = "start_command_execution"
    description: str = "Start long-running commands asynchronously"
    args_schema: type[BaseModel] = StartCommandInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, command: str, thread_id: str = "default") -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # Start command execution
            response = code_interpreter.invoke(
                method="startCommandExecution", params={"command": command}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error starting command: {e!s}"

    async def _arun(self, command: str, thread_id: str = "default") -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(command=command, thread_id=thread_id)


class GetTaskTool(BaseTool):
    """Tool for checking status of async tasks."""

    name: str = "get_task"
    description: str = "Check status of async tasks"
    args_schema: type[BaseModel] = GetTaskInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, task_id: str, thread_id: str = "default") -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # Get task status
            response = code_interpreter.invoke(
                method="getTask", params={"taskId": task_id}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error getting task status: {e!s}"

    async def _arun(self, task_id: str, thread_id: str = "default") -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(task_id=task_id, thread_id=thread_id)


class StopTaskTool(BaseTool):
    """Tool for stopping running tasks."""

    name: str = "stop_task"
    description: str = "Stop running tasks"
    args_schema: type[BaseModel] = StopTaskInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, task_id: str, thread_id: str = "default") -> str:
        try:
            # Get or create code interpreter
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )

            # Stop task
            response = code_interpreter.invoke(
                method="stopTask", params={"taskId": task_id}
            )

            return extract_output_from_stream(response)
        except Exception as e:
            return f"Error stopping task: {e!s}"

    async def _arun(self, task_id: str, thread_id: str = "default") -> str:
        # Use _run as we're working with a synchronous API that's thread-safe
        return self._run(task_id=task_id, thread_id=thread_id)


class UploadFileInput(BaseModel):
    """Input for UploadFile."""

    path: str = Field(description="Relative path where the file should be saved")
    content: str = Field(description="File content as a string")
    description: str = Field(
        default="", description="Optional semantic description of the file contents"
    )
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class UploadFilesInput(BaseModel):
    """Input for UploadFiles."""

    files: list[dict[str, str]] = Field(
        description="List of dicts with 'path', 'content', and optional 'description'"
    )
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class InstallPackagesInput(BaseModel):
    """Input for InstallPackages."""

    packages: list[str] = Field(description="List of package names to install via pip")
    upgrade: bool = Field(
        default=False, description="Whether to add --upgrade flag"
    )
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class DownloadFileInput(BaseModel):
    """Input for DownloadFile."""

    path: str = Field(description="Path to the file to download")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class DownloadFilesInput(BaseModel):
    """Input for DownloadFiles."""

    paths: list[str] = Field(description="List of file paths to download")
    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class ClearContextInput(BaseModel):
    """Input for ClearContext."""

    thread_id: str = Field(
        default="default", description="Thread ID for the code interpreter session"
    )


class UploadFileTool(BaseTool):
    """Tool for uploading a file with an optional semantic description."""

    name: str = "upload_file"
    description: str = "Upload a file to the code interpreter environment with an optional description"
    args_schema: type[BaseModel] = UploadFileInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(
        self,
        path: str,
        content: str,
        description: str = "",
        thread_id: str = "default",
    ) -> str:
        try:
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )
            result = code_interpreter.upload_file(
                path=path, content=content, description=description
            )
            return extract_output_from_stream(result)
        except Exception as e:
            logger.error("Failed to upload file: %s", e, exc_info=True)
            return f"Error uploading file: {e!s}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class UploadFilesTool(BaseTool):
    """Tool for uploading multiple files at once."""

    name: str = "upload_files"
    description: str = "Upload multiple files to the code interpreter environment"
    args_schema: type[BaseModel] = UploadFilesInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(
        self,
        files: list[dict[str, str]],
        thread_id: str = "default",
    ) -> str:
        try:
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )
            result = code_interpreter.upload_files(files=files)
            return extract_output_from_stream(result)
        except Exception as e:
            logger.error("Failed to upload files: %s", e, exc_info=True)
            return f"Error uploading files: {e!s}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class InstallPackagesTool(BaseTool):
    """Tool for installing Python packages via pip."""

    name: str = "install_packages"
    description: str = "Install Python packages via pip in the code interpreter environment"
    args_schema: type[BaseModel] = InstallPackagesInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(
        self,
        packages: list[str],
        upgrade: bool = False,
        thread_id: str = "default",
    ) -> str:
        try:
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )
            result = code_interpreter.install_packages(
                packages=packages, upgrade=upgrade
            )
            return extract_output_from_stream(result)
        except Exception as e:
            logger.error("Failed to install packages: %s", e, exc_info=True)
            return f"Error installing packages: {e!s}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class DownloadFileTool(BaseTool):
    """Tool for downloading a file from the sandbox."""

    name: str = "download_file"
    description: str = "Download a file from the code interpreter environment"
    args_schema: type[BaseModel] = DownloadFileInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, path: str, thread_id: str = "default") -> str:
        try:
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )
            content = code_interpreter.download_file(path=path)
            if isinstance(content, bytes):
                return f"[Binary file: {len(content)} bytes]"
            return content
        except Exception as e:
            logger.error("Failed to download file: %s", e, exc_info=True)
            return f"Error downloading file: {e!s}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class DownloadFilesTool(BaseTool):
    """Tool for downloading multiple files from the sandbox."""

    name: str = "download_files"
    description: str = "Download multiple files from the code interpreter environment"
    args_schema: type[BaseModel] = DownloadFilesInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, paths: list[str], thread_id: str = "default") -> str:
        try:
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )
            files = code_interpreter.download_files(paths=paths)
            results = []
            for file_path, content in files.items():
                if isinstance(content, bytes):
                    results.append(f"==== {file_path} [Binary: {len(content)} bytes] ====")
                else:
                    results.append(f"==== {file_path} ====\n{content}")
            return "\n".join(results) if results else "No files found"
        except Exception as e:
            logger.error("Failed to download files: %s", e, exc_info=True)
            return f"Error downloading files: {e!s}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class ClearContextTool(BaseTool):
    """Tool for clearing all variable state in the Python execution context."""

    name: str = "clear_context"
    description: str = "Clear all variable state in the Python execution context"
    args_schema: type[BaseModel] = ClearContextInput
    toolkit: Any = Field(default=None, exclude=True)

    def __init__(self, toolkit):
        super().__init__()
        self.toolkit = toolkit

    def _run(self, thread_id: str = "default") -> str:
        try:
            code_interpreter = self.toolkit._get_or_create_interpreter(
                thread_id=thread_id
            )
            code_interpreter.clear_context()
            return "Python execution context cleared successfully"
        except Exception as e:
            logger.error("Failed to clear context: %s", e, exc_info=True)
            return f"Error clearing context: {e!s}"

    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)


class CodeInterpreterToolkit:
    """Toolkit for working with AWS Bedrock code interpreter environment.

    This toolkit provides a set of tools for working with a remote code interpreter environment:

    * execute_code - Run code in various languages (primarily Python)
    * execute_command - Run shell commands
    * read_files - Read content of files in the environment
    * list_files - List files in directories
    * delete_files - Remove files from the environment
    * write_files - Create or update files
    * start_command_execution - Start long-running commands asynchronously
    * get_task - Check status of async tasks
    * stop_task - Stop running tasks
    * upload_file - Upload a single file to the environment
    * upload_files - Upload multiple files to the environment
    * install_packages - Install pip packages in the environment
    * download_file - Download a single file from the environment
    * download_files - Download multiple files from the environment
    * clear_context - Reset the Python execution context

    The toolkit lazily initializes the code interpreter session on first use.
    It supports multiple threads by maintaining separate code interpreter sessions for each thread ID.

    Example:
        ```python
        from crewai import Agent, Task, Crew
        from crewai_tools.aws.bedrock.code_interpreter import (
            create_code_interpreter_toolkit,
        )

        # Create the code interpreter toolkit
        toolkit, code_tools = create_code_interpreter_toolkit(region="us-west-2")

        # Create a CrewAI agent that uses the code interpreter tools
        developer_agent = Agent(
            role="Python Developer",
            goal="Create and execute Python code to solve problems",
            backstory="You're a skilled Python developer with expertise in data analysis.",
            tools=code_tools,
        )

        # Create a task for the agent
        coding_task = Task(
            description="Write a Python function that calculates the factorial of a number and test it.",
            agent=developer_agent,
        )

        # Create and run the crew
        crew = Crew(agents=[developer_agent], tasks=[coding_task])
        result = crew.kickoff()

        # Clean up resources when done
        import asyncio

        asyncio.run(toolkit.cleanup())
        ```
    """

    def __init__(
        self,
        region: str = "us-west-2",
        identifier: str | None = None,
        session_timeout_seconds: int | None = None,
    ):
        """Initialize the toolkit.

        Args:
            region: AWS region for the code interpreter
            identifier: Code interpreter sandbox identifier. Defaults to the system interpreter.
                Use a custom ID from create_code_interpreter() for custom configurations.
            session_timeout_seconds: Session timeout in seconds. Defaults to 900
                (15 minutes) if not specified.
        """
        self.region = region
        self.identifier = identifier
        self.session_timeout_seconds = session_timeout_seconds
        self._code_interpreters: dict[str, CodeInterpreter] = {}
        self.tools: list[BaseTool] = []
        self._setup_tools()

    def _setup_tools(self) -> None:
        """Initialize tools without creating any code interpreter sessions."""
        self.tools = [
            ExecuteCodeTool(self),
            ExecuteCommandTool(self),
            ReadFilesTool(self),
            ListFilesTool(self),
            DeleteFilesTool(self),
            WriteFilesTool(self),
            StartCommandTool(self),
            GetTaskTool(self),
            StopTaskTool(self),
            UploadFileTool(self),
            UploadFilesTool(self),
            InstallPackagesTool(self),
            DownloadFileTool(self),
            DownloadFilesTool(self),
            ClearContextTool(self),
        ]

    def _get_or_create_interpreter(self, thread_id: str = "default") -> CodeInterpreter:
        """Get or create a code interpreter for the specified thread.

        Args:
            thread_id: Thread ID for the code interpreter session

        Returns:
            CodeInterpreter instance
        """
        if thread_id in self._code_interpreters:
            return self._code_interpreters[thread_id]

        # Create a new code interpreter for this thread
        from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter

        code_interpreter = CodeInterpreter(
            region=self.region,
            integration_source="crewai",
        )
        start_kwargs: dict[str, Any] = {}
        if self.identifier is not None:
            start_kwargs["identifier"] = self.identifier
        if self.session_timeout_seconds is not None:
            start_kwargs["session_timeout_seconds"] = self.session_timeout_seconds
        code_interpreter.start(**start_kwargs)
        logger.info(
            f"Started code interpreter with session_id:{code_interpreter.session_id} for thread:{thread_id}"
        )

        # Store the interpreter
        self._code_interpreters[thread_id] = code_interpreter
        return code_interpreter

    def get_tools(self) -> list[BaseTool]:
        """Get the list of code interpreter tools.

        Returns:
            List of CrewAI tools
        """
        return self.tools

    def get_tools_by_name(self) -> dict[str, BaseTool]:
        """Get a dictionary of tools mapped by their names.

        Returns:
            Dictionary of {tool_name: tool}
        """
        return {tool.name: tool for tool in self.tools}

    async def cleanup(self, thread_id: str | None = None) -> None:
        """Clean up resources.

        Args:
            thread_id: Optional thread ID to clean up. If None, cleans up all sessions.
        """
        if thread_id:
            # Clean up a specific thread's session
            if thread_id in self._code_interpreters:
                try:
                    self._code_interpreters[thread_id].stop()
                    del self._code_interpreters[thread_id]
                    logger.info(
                        f"Code interpreter session for thread {thread_id} cleaned up"
                    )
                except Exception as e:
                    logger.warning(
                        f"Error stopping code interpreter for thread {thread_id}: {e}"
                    )
        else:
            # Clean up all sessions
            thread_ids = list(self._code_interpreters.keys())
            for tid in thread_ids:
                try:
                    self._code_interpreters[tid].stop()
                except Exception as e:  # noqa: PERF203
                    logger.warning(
                        f"Error stopping code interpreter for thread {tid}: {e}"
                    )

            self._code_interpreters = {}
            logger.info("All code interpreter sessions cleaned up")


def create_code_interpreter_toolkit(
    region: str = "us-west-2",
    identifier: str | None = None,
    session_timeout_seconds: int | None = None,
) -> tuple[CodeInterpreterToolkit, list[BaseTool]]:
    """Create a CodeInterpreterToolkit.

    Args:
        region: AWS region for code interpreter
        identifier: Code interpreter sandbox identifier. Defaults to the system interpreter.
            Use a custom ID from create_code_interpreter() for custom configurations.
        session_timeout_seconds: Session timeout in seconds. Defaults to 900
            (15 minutes) if not specified.

    Returns:
        Tuple of (toolkit, tools)
    """
    toolkit = CodeInterpreterToolkit(
        region=region,
        identifier=identifier,
        session_timeout_seconds=session_timeout_seconds,
    )
    tools = toolkit.get_tools()
    return toolkit, tools
