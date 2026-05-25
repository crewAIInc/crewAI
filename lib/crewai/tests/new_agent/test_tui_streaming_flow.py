"""Test that simulates the TUI's exact streaming flow to detect empty bubble bug."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from crewai.new_agent import NewAgent, AgentSettings
from crewai.tools import BaseTool


def _make_tool_call(name: str, args: str = "{}", call_id: str = "call_1"):
    fn = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(function=fn, id=call_id, type="function")


class MockListFiles(BaseTool):
    name: str = "list_files_in_directory"
    description: str = "List files in a directory"

    def _run(self, path: str = ".") -> str:
        return "main.py\nutils.py\nREADME.md"


class MockReadFile(BaseTool):
    name: str = "read_a_files_content"
    description: str = "Read file content"

    def _run(self, path: str = "") -> str:
        return "from flask import Flask\napp = Flask(__name__)"


class TestTUIStreamingFlow:
    """Simulate the TUI's exact streaming loop to see if content is lost."""

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_tui_flow_single_tool(self, mock_llm_response):
        """Simulate TUI flow: stream with one tool call."""
        tool_call = _make_tool_call("list_files_in_directory", '{"path": "."}')
        mock_llm_response.side_effect = [
            [tool_call],
            "I found these files: main.py, utils.py, and README.md in the current directory.",
        ]

        agent = NewAgent(
            role="Coder",
            goal="Read files",
            backstory="Expert.",
            tools=[MockListFiles()],
            settings=AgentSettings(planning_enabled=False),
        )

        # Simulate TUI streaming loop
        _TOOL_RESET = "\x00TOOL_RESET\x00"
        bubble_created = False
        bubble_content = ""
        accumulated = ""
        stream_chars = 0
        tool_reset_count = 0
        chunk_count = 0
        stream_error = None

        stream = agent.stream("What files are here?")
        while True:
            try:
                chunk = await asyncio.wait_for(anext(stream), timeout=30.0)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                accumulated += "\n\n[Response timed out]"
                stream_error = "timeout"
                break
            except Exception as err:
                stream_error = str(err)
                break

            if chunk == _TOOL_RESET:
                accumulated = ""
                stream_chars = 0
                tool_reset_count += 1
                continue

            chunk_count += 1
            accumulated += chunk
            stream_chars += len(chunk)

            if not bubble_created:
                bubble_created = True
                bubble_content = accumulated
            else:
                bubble_content = accumulated

        # Post-streaming: get response (same as TUI)
        response = agent.last_stream_result

        final_content = accumulated
        if response and response.content:
            final_content = response.content

        # Last resort: check conversation history
        if not final_content:
            history = agent.conversation_history
            if history and history[-1].role == "agent":
                final_content = history[-1].content or ""

        print(f"\n--- TUI Flow (single tool) ---")
        print(f"  stream_error:    {stream_error}")
        print(f"  tool_resets:     {tool_reset_count}")
        print(f"  chunk_count:     {chunk_count}")
        print(f"  accumulated_len: {len(accumulated)}")
        print(f"  accumulated:     {accumulated[:100]!r}")
        print(f"  bubble_created:  {bubble_created}")
        print(f"  bubble_content:  {bubble_content[:100]!r}")
        print(f"  response:        {response is not None}")
        _rc = response.content[:100] if response and response.content else "None"
        print(f"  response.content:{_rc!r}")
        print(f"  final_content:   {final_content[:100]!r}")

        assert stream_error is None, f"Stream error: {stream_error}"
        assert final_content, "final_content should not be empty!"
        assert response is not None, "response should not be None"
        assert response.content, "response.content should not be empty"

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_tui_flow_multi_tool(self, mock_llm_response):
        """Simulate TUI flow: stream with two sequential tool calls."""
        tc1 = _make_tool_call("list_files_in_directory", '{"path": "."}', "call_1")
        tc2 = _make_tool_call("read_a_files_content", '{"path": "main.py"}', "call_2")
        mock_llm_response.side_effect = [
            [tc1],
            [tc2],
            "I analyzed the project. main.py contains a Flask app with routes.",
        ]

        agent = NewAgent(
            role="Coder",
            goal="Read files",
            backstory="Expert.",
            tools=[MockListFiles(), MockReadFile()],
            settings=AgentSettings(planning_enabled=False),
        )

        _TOOL_RESET = "\x00TOOL_RESET\x00"
        bubble_created = False
        bubble_content = ""
        accumulated = ""
        stream_chars = 0
        tool_reset_count = 0
        chunk_count = 0
        stream_error = None

        stream = agent.stream("Analyze the project")
        while True:
            try:
                chunk = await asyncio.wait_for(anext(stream), timeout=30.0)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                accumulated += "\n\n[Response timed out]"
                stream_error = "timeout"
                break
            except Exception as err:
                stream_error = str(err)
                break

            if chunk == _TOOL_RESET:
                accumulated = ""
                stream_chars = 0
                tool_reset_count += 1
                continue

            chunk_count += 1
            accumulated += chunk
            stream_chars += len(chunk)

            if not bubble_created:
                bubble_created = True
            bubble_content = accumulated

        response = agent.last_stream_result

        final_content = accumulated
        if response and response.content:
            final_content = response.content

        if not final_content:
            history = agent.conversation_history
            if history and history[-1].role == "agent":
                final_content = history[-1].content or ""

        print(f"\n--- TUI Flow (multi tool) ---")
        print(f"  stream_error:    {stream_error}")
        print(f"  tool_resets:     {tool_reset_count}")
        print(f"  chunk_count:     {chunk_count}")
        print(f"  accumulated_len: {len(accumulated)}")
        print(f"  bubble_created:  {bubble_created}")
        print(f"  response:        {response is not None}")
        _rc = response.content[:100] if response and response.content else "None"
        print(f"  response.content:{_rc!r}")
        print(f"  final_content:   {final_content[:100]!r}")

        assert stream_error is None, f"Stream error: {stream_error}"
        assert final_content, "final_content should not be empty!"
        assert response is not None
        assert response.content

    @patch("crewai.new_agent.executor.aget_llm_response")
    @pytest.mark.asyncio
    async def test_tui_flow_error_during_response(self, mock_llm_response):
        """Simulate TUI flow: LLM raises error after tool use."""
        tc1 = _make_tool_call("list_files_in_directory", '{"path": "."}', "call_1")
        mock_llm_response.side_effect = [
            [tc1],
            ValueError("Context length exceeded"),
        ]

        agent = NewAgent(
            role="Coder",
            goal="Read files",
            backstory="Expert.",
            tools=[MockListFiles()],
            settings=AgentSettings(planning_enabled=False),
        )

        _TOOL_RESET = "\x00TOOL_RESET\x00"
        accumulated = ""
        stream_error = None

        stream = agent.stream("What files?")
        while True:
            try:
                chunk = await asyncio.wait_for(anext(stream), timeout=30.0)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                accumulated += "\n\n[Response timed out]"
                stream_error = "timeout"
                break
            except Exception as err:
                stream_error = str(err)
                break

            if chunk == _TOOL_RESET:
                accumulated = ""
                continue

            accumulated += chunk

        response = agent.last_stream_result

        final_content = accumulated
        if response and response.content:
            final_content = response.content

        if not final_content:
            history = agent.conversation_history
            if history and history[-1].role == "agent":
                final_content = history[-1].content or ""

        print(f"\n--- TUI Flow (error) ---")
        print(f"  stream_error:    {stream_error}")
        print(f"  accumulated_len: {len(accumulated)}")
        print(f"  accumulated:     {accumulated[:200]!r}")
        print(f"  response:        {response is not None}")
        _rc = response.content[:200] if response and response.content else "None"
        print(f"  response.content:{_rc!r}")
        print(f"  final_content:   {final_content[:200]!r}")

        # With our fix, even errors should produce SOME content
        assert response is not None, "response should not be None even on error"
        assert final_content, "final_content should not be empty even on error"
        print(f"  [OK] Error handled gracefully, user sees: {final_content[:100]!r}")
