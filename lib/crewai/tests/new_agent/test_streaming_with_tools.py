"""Test streaming behavior when tools are used — the core issue is
whether a final text response comes back after tool execution."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.new_agent import NewAgent, AgentSettings, Message


def _make_tool_call(name: str, args: str = "{}", call_id: str = "call_1"):
    """Return a mock tool-call object matching OpenAI format."""
    fn = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(function=fn, id=call_id, type="function")


class TestStreamingWithTools:
    """Test that streaming yields a final response after tool use."""

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_amessage_with_tool_calls(self, mock_llm_response):
        """ainvoke should return a message with content after tool use."""
        # First call: LLM returns tool calls
        tool_call = _make_tool_call("list_files_in_directory", '{"path": "."}')
        # Second call: LLM returns final text
        mock_llm_response.side_effect = [
            [tool_call],          # iteration 0: tool call
            "Here are the files I found: main.py, utils.py",  # iteration 1: final response
        ]

        from crewai.tools import BaseTool
        from pydantic import Field

        class MockListFiles(BaseTool):
            name: str = "list_files_in_directory"
            description: str = "List files in a directory"

            def _run(self, path: str = ".") -> str:
                return "main.py\nutils.py\nREADME.md"

        agent = NewAgent(
            role="Coder",
            goal="Read files",
            backstory="Expert.",
            tools=[MockListFiles()],
            settings=AgentSettings(planning_enabled=False),
        )

        response = await agent.amessage("What files are in the project?")

        assert response.role == "agent"
        assert response.content  # Must be non-empty!
        assert "files" in response.content.lower() or "main.py" in response.content
        assert len(agent.conversation_history) == 2
        print(f"\n[ainvoke result] content={response.content!r}")
        print(f"[ainvoke result] tools_used={response.tools_used}")

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, mock_llm_response):
        """stream() should yield the final response text after tool use."""
        tool_call = _make_tool_call("list_files_in_directory", '{"path": "."}')
        mock_llm_response.side_effect = [
            [tool_call],
            "Here are the files I found: main.py, utils.py",
        ]

        from crewai.tools import BaseTool

        class MockListFiles(BaseTool):
            name: str = "list_files_in_directory"
            description: str = "List files in a directory"

            def _run(self, path: str = ".") -> str:
                return "main.py\nutils.py\nREADME.md"

        agent = NewAgent(
            role="Coder",
            goal="Read files",
            backstory="Expert.",
            tools=[MockListFiles()],
            settings=AgentSettings(planning_enabled=False),
        )

        chunks: list[str] = []
        tool_resets = 0
        _TOOL_RESET = "\x00TOOL_RESET\x00"

        async for chunk in agent.stream("What files are in the project?"):
            if chunk == _TOOL_RESET:
                tool_resets += 1
                continue
            chunks.append(chunk)

        all_text = "".join(chunks)
        result = agent.last_stream_result

        print(f"\n[stream] tool_resets={tool_resets}")
        print(f"[stream] total chunks={len(chunks)}")
        print(f"[stream] all_text={all_text!r}")
        _rc = result.content if result else "None"
        print(f"[stream] last_stream_result.content={_rc!r}")

        assert result is not None, "last_stream_result should be set"
        assert result.content, f"result.content should be non-empty, got: {result.content!r}"
        # Either chunks contain text, or the fallback yield fired
        assert all_text or result.content, "Should have content from stream or result"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_stream_with_multiple_tool_calls(self, mock_llm_response):
        """Multiple sequential tool calls should still result in a final response."""
        tc1 = _make_tool_call("list_files_in_directory", '{"path": "."}', "call_1")
        tc2 = _make_tool_call("read_a_files_content", '{"path": "main.py"}', "call_2")
        mock_llm_response.side_effect = [
            [tc1],           # iteration 0: list files
            [tc2],           # iteration 1: read file
            "I read the file. It contains a Flask app with 3 routes.",  # iteration 2: final
        ]

        from crewai.tools import BaseTool

        class MockListFiles(BaseTool):
            name: str = "list_files_in_directory"
            description: str = "List files"

            def _run(self, path: str = ".") -> str:
                return "main.py"

        class MockReadFile(BaseTool):
            name: str = "read_a_files_content"
            description: str = "Read file content"

            def _run(self, path: str = "") -> str:
                return "from flask import Flask\napp = Flask(__name__)"

        agent = NewAgent(
            role="Coder",
            goal="Read files",
            backstory="Expert.",
            tools=[MockListFiles(), MockReadFile()],
            settings=AgentSettings(planning_enabled=False),
        )

        chunks: list[str] = []
        tool_resets = 0
        _TOOL_RESET = "\x00TOOL_RESET\x00"

        async for chunk in agent.stream("Analyze the project"):
            if chunk == _TOOL_RESET:
                tool_resets += 1
                continue
            chunks.append(chunk)

        all_text = "".join(chunks)
        result = agent.last_stream_result

        print(f"\n[multi-tool stream] tool_resets={tool_resets}")
        print(f"[multi-tool stream] total chunks={len(chunks)}")
        print(f"[multi-tool stream] all_text={all_text!r}")
        _rc2 = result.content if result else "None"
        print(f"[multi-tool stream] result.content={_rc2!r}")

        assert result is not None
        assert result.content, f"result.content should be non-empty, got: {result.content!r}"
        assert tool_resets >= 1, "Should have at least one TOOL_RESET"
