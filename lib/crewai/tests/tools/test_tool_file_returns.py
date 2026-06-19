"""Tests for tools returning ``FileInput`` instances.

Verifies that when a tool's ``_run`` method returns a ``BaseFile`` instance
(or a collection of them), the framework:
  1. Detects the file(s) via ``extract_files_from_tool_result``.
  2. Replaces the raw return with a confirmation message string.
  3. Propagates the files through ``ToolResult.files``.
  4. Attaches the files to the agent's message history.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from crewai_files import File, ImageFile, TextFile

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentAction
from crewai.tools import BaseTool
from crewai.tools.tool_calling import ToolCalling
from crewai.tools.tool_types import ToolResult
from crewai.tools.tool_usage import ToolUsage
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Fixture tools
# ---------------------------------------------------------------------------


class SingleFileToolInput(BaseModel):
    path: str = Field(description="Path to get file from")


class SingleFileTool(BaseTool):
    name: str = "Get Document"
    description: str = "Returns a single file"
    args_schema: type[BaseModel] = SingleFileToolInput

    def _run(self, path: str, **kwargs) -> TextFile:
        return TextFile(source=b"hello world")


class MultiFileToolInput(BaseModel):
    query: str = Field(description="Query to search for")


class MultiFileTool(BaseTool):
    name: str = "Search Files"
    description: str = "Returns multiple files"
    args_schema: type[BaseModel] = MultiFileToolInput

    def _run(self, query: str, **kwargs) -> list[TextFile]:
        return [
            TextFile(source=b"file one"),
            TextFile(source=b"file two"),
        ]


class DictFileToolInput(BaseModel):
    name: str = Field(description="Name")


class DictFileTool(BaseTool):
    name: str = "Get Named Files"
    description: str = "Returns a dict of named files"
    args_schema: type[BaseModel] = DictFileToolInput

    def _run(self, name: str, **kwargs) -> dict[str, TextFile]:
        return {
            "notes": TextFile(source=b"notes content"),
            "report": TextFile(source=b"report content"),
        }


class RegularTool(BaseTool):
    name: str = "Regular Tool"
    description: str = "Returns a string"

    def _run(self, **kwargs) -> str:
        return "just a string"


# ---------------------------------------------------------------------------
# Tests: ToolUsage file extraction
# ---------------------------------------------------------------------------


class TestToolUsageFileExtraction:
    """Test that ToolUsage._use extracts files from tool returns."""

    def _make_tool_usage(self, tool: BaseTool) -> ToolUsage:
        structured = tool.to_structured_tool()
        mock_agent = MagicMock()
        mock_agent.key = "agent_key"
        mock_agent.role = "agent_role"
        mock_agent._original_role = "agent_role"
        mock_agent.verbose = False
        mock_agent.fingerprint = None
        mock_agent.tools_results = []

        mock_task = MagicMock()
        mock_task.delegations = 0
        mock_task.name = "Test"
        mock_task.description = "Test"
        mock_task.id = "t-id"

        mock_action = MagicMock()
        mock_action.tool = tool.name
        mock_action.tool_input = "{}"

        return ToolUsage(
            tools_handler=MagicMock(cache=None, last_used_tool=None),
            tools=[structured],
            task=mock_task,
            function_calling_llm=None,
            agent=mock_agent,
            action=mock_action,
        )

    def test_single_file_tool_extracts_files(self) -> None:
        """When a tool returns a single BaseFile, _last_extracted_files is set."""
        tool = SingleFileTool()
        tool_usage = self._make_tool_usage(tool)
        calling = ToolCalling(tool_name="get_document", arguments={"path": "/a"})

        result = tool_usage.use(calling=calling, tool_string="Action: get_document")

        assert tool_usage._last_extracted_files is not None
        assert len(tool_usage._last_extracted_files) == 1
        assert "Added 1 file" in result

    def test_multi_file_tool_extracts_files(self) -> None:
        """When a tool returns a list of BaseFile, all are extracted."""
        tool = MultiFileTool()
        tool_usage = self._make_tool_usage(tool)
        calling = ToolCalling(tool_name="search_files", arguments={"query": "q"})

        result = tool_usage.use(calling=calling, tool_string="Action: search_files")

        assert tool_usage._last_extracted_files is not None
        assert len(tool_usage._last_extracted_files) == 2
        assert "Added 2 files" in result

    def test_dict_file_tool_extracts_files(self) -> None:
        """When a tool returns a dict of BaseFile, keys are preserved."""
        tool = DictFileTool()
        tool_usage = self._make_tool_usage(tool)
        calling = ToolCalling(tool_name="get_named_files", arguments={"name": "test"})

        result = tool_usage.use(calling=calling, tool_string="Action: get_named_files")

        assert tool_usage._last_extracted_files is not None
        assert "notes" in tool_usage._last_extracted_files
        assert "report" in tool_usage._last_extracted_files

    def test_regular_tool_no_files_extracted(self) -> None:
        """Regular string-returning tools don't trigger file extraction."""
        tool = RegularTool()
        tool_usage = self._make_tool_usage(tool)
        calling = ToolCalling(tool_name="regular_tool", arguments={})

        result = tool_usage.use(calling=calling, tool_string="Action: regular_tool")

        assert tool_usage._last_extracted_files is None
        assert "just a string" in result


# ---------------------------------------------------------------------------
# Tests: execute_tool_and_check_finality propagates files
# ---------------------------------------------------------------------------


class TestToolUtilsFilePropagation:
    """Test that execute_tool_and_check_finality propagates files in ToolResult."""

    def test_file_tool_returns_tool_result_with_files(self) -> None:
        """ToolResult.files should be set when tool returns a file."""
        tool = SingleFileTool()
        structured = tool.to_structured_tool()
        action = AgentAction(
            thought="Need a doc",
            tool="get_document",
            tool_input='{"path": "/a"}',
            text='Action: get_document\nAction Input: {"path": "/a"}',
        )

        result = execute_tool_and_check_finality(
            agent_action=action,
            tools=[structured],
            agent_key="k",
            agent_role="r",
        )

        assert isinstance(result, ToolResult)
        assert result.files is not None
        assert len(result.files) == 1
        assert "Added 1 file" in result.result

    def test_regular_tool_returns_tool_result_without_files(self) -> None:
        """ToolResult.files should be None for regular tools."""
        tool = RegularTool()
        structured = tool.to_structured_tool()
        action = AgentAction(
            thought="Run a tool",
            tool="regular_tool",
            tool_input="{}",
            text="Action: regular_tool\nAction Input: {}",
        )

        result = execute_tool_and_check_finality(
            agent_action=action,
            tools=[structured],
            agent_key="k",
            agent_role="r",
        )

        assert isinstance(result, ToolResult)
        assert result.files is None
        assert "just a string" in result.result


# ---------------------------------------------------------------------------
# Tests: CrewAgentExecutor._attach_tool_files_to_messages
# ---------------------------------------------------------------------------


class TestAttachToolFilesToMessages:
    """Test that _attach_tool_files_to_messages correctly attaches files."""

    def _make_executor(self, messages: list[dict]) -> CrewAgentExecutor:
        """Create a minimal executor with pre-set messages.

        Uses ``model_construct`` to skip pydantic validation so tests can
        exercise ``_attach_tool_files_to_messages`` without supplying a
        complete agent/task wiring.
        """
        executor = CrewAgentExecutor.model_construct(messages=messages)
        return executor

    def test_attaches_files_to_last_user_message(self) -> None:
        """Files are merged into the most recent user message."""
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "working..."},
        ]
        executor = self._make_executor(messages)
        test_file = TextFile(source=b"content")
        executor._attach_tool_files_to_messages({"doc": test_file})

        user_msg = messages[1]
        assert "files" in user_msg
        assert user_msg["files"]["doc"] is test_file

    def test_merges_with_existing_files(self) -> None:
        """New files are merged into existing files on the user message."""
        existing_file = ImageFile(source=b"\x89PNG\r\n\x1a\n")
        messages = [
            {"role": "user", "content": "go", "files": {"img": existing_file}},
        ]
        executor = self._make_executor(messages)
        new_file = TextFile(source=b"text")
        executor._attach_tool_files_to_messages({"doc": new_file})

        user_msg = messages[0]
        assert user_msg["files"]["img"] is existing_file
        assert user_msg["files"]["doc"] is new_file

    def test_creates_user_message_if_none_exists(self) -> None:
        """If no user message exists, a new one is appended."""
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "assistant", "content": "answer"},
        ]
        executor = self._make_executor(messages)
        test_file = TextFile(source=b"content")
        executor._attach_tool_files_to_messages({"doc": test_file})

        assert len(messages) == 3
        user_msg = messages[-1]
        assert user_msg["role"] == "user"
        assert user_msg["files"]["doc"] is test_file

    def test_no_op_for_empty_files(self) -> None:
        """Empty files dict doesn't modify messages."""
        messages = [{"role": "user", "content": "hi"}]
        executor = self._make_executor(messages)
        executor._attach_tool_files_to_messages({})

        assert "files" not in messages[0]


# ---------------------------------------------------------------------------
# Tests: _handle_agent_action with files
# ---------------------------------------------------------------------------


class TestHandleAgentActionWithFiles:
    """Test that _handle_agent_action calls _attach_tool_files_to_messages."""

    def test_handle_agent_action_attaches_files(self) -> None:
        """When ToolResult has files, they get attached to messages."""
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
        ]

        executor = CrewAgentExecutor.model_construct(
            messages=messages,
            step_callback=None,
            task=None,
            crew=None,
        )

        action = AgentAction(
            thought="Run tool",
            tool="tool",
            tool_input="{}",
            text="Action: tool\nAction Input: {}",
        )
        test_file = TextFile(source=b"data")
        tool_result = ToolResult(
            result="Added 1 file to the agent context: 'doc' (text/plain).",
            result_as_answer=False,
            files={"doc": test_file},
        )

        with patch.object(executor, "_show_logs"):
            executor._handle_agent_action(action, tool_result)

        user_msg = messages[1]
        assert "files" in user_msg
        assert user_msg["files"]["doc"] is test_file
