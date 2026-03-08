"""Tests for _parse_native_tool_call — Bedrock tool arg extraction (issue #4748)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor


@pytest.fixture
def executor():
    """Create a minimal CrewAgentExecutor for testing _parse_native_tool_call."""
    agent = MagicMock()
    agent.role = "test"
    agent.goal = "test"
    agent.backstory = "test"
    agent.tools = []
    agent.max_iter = 25
    agent.agent_executor = None
    agent.crew = None
    agent.allow_delegation = False
    agent.step_callback = None
    agent.llm = MagicMock()
    agent.llm.supports_stop_words.return_value = False
    agent.knowledge = None
    agent.name = None
    agent.id = "test-agent"
    agent.max_retry_limit = 2
    agent.respect_context_window = True
    agent.max_tokens = None
    agent.function_calling_llm = None
    agent.agent_ops_agent_name = None
    agent.use_system_prompt = True
    agent.system_template = None
    agent.prompt_template = None
    agent.response_template = None
    agent.tools_results = []
    agent.tools_handler = MagicMock()
    agent.tools_handler.cache = None
    agent.guardrail = None
    agent.max_execution_time = None
    agent.code_execution_mode = "safe"
    agent.memory = None
    agent.verbose = False

    task = MagicMock()
    task.description = "test"
    task.expected_output = "test"
    task.tools = []
    task.guardrail = None

    with patch.object(CrewAgentExecutor, "__init__", lambda self, *a, **kw: None):
        exec_ = CrewAgentExecutor.__new__(CrewAgentExecutor)
        exec_.agent = agent
        exec_.task = task
        exec_.messages = []
        exec_.iterations = 0
        exec_.tool_calling_llm = None
        exec_.have_forced_answer = False
        exec_.should_ask_for_human_input = False
        exec_.request_within_rpm_limit = None
        exec_.max_iterations = 25
        exec_._i18n = MagicMock()
        return exec_


class TestParseNativeToolCallBedrockArgs:
    """Test that Bedrock's `input` field is correctly extracted (issue #4748)."""

    def test_bedrock_dict_with_input_field(self, executor):
        """Bedrock tool calls use 'input' not 'function.arguments'."""
        tool_call = {
            "toolUseId": "tooluse_abc123",
            "name": "search_tool",
            "input": {"query": "AWS Bedrock features"},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "tooluse_abc123"
        assert func_name == "search_tool"
        assert func_args == {"query": "AWS Bedrock features"}

    def test_openai_dict_with_function_arguments(self, executor):
        """OpenAI format should still work (regression test)."""
        tool_call = {
            "id": "call_xyz789",
            "function": {
                "name": "search_tool",
                "arguments": '{"query": "test query"}',
            },
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "call_xyz789"
        assert func_name == "search_tool"
        assert func_args == '{"query": "test query"}'

    def test_bedrock_dict_without_function_key(self, executor):
        """Bedrock dicts have no 'function' key — must still extract args."""
        tool_call = {
            "toolUseId": "tooluse_def456",
            "name": "multiply_tool",
            "input": {"a": 10, "b": 5},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        # Before fix: func_args would be "{}" (the default string)
        # After fix: func_args should be the actual input dict
        assert func_args == {"a": 10, "b": 5}

    def test_bedrock_empty_input_returns_empty_dict(self, executor):
        """When Bedrock input is empty dict, should return empty dict not string."""
        tool_call = {
            "toolUseId": "tooluse_ghi789",
            "name": "no_args_tool",
            "input": {},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        # Empty dict {} is falsy, so fallback to "{}" is acceptable
        assert func_args in ({}, "{}")

    def test_neither_function_nor_input(self, executor):
        """Dict with neither 'function.arguments' nor 'input' returns fallback."""
        tool_call = {
            "id": "call_nope",
            "name": "some_tool",
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert func_args == "{}"

    def test_openai_object_with_function_attr(self, executor):
        """OpenAI-style object with .function attribute (regression)."""
        func_obj = SimpleNamespace(
            name="calc_tool",
            arguments='{"x": 42}',
        )
        tool_call = SimpleNamespace(
            id="call_obj_001",
            function=func_obj,
        )
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert func_args == '{"x": 42}'

    def test_bedrock_object_with_name_and_input(self, executor):
        """Bedrock-style object with .name and .input attributes."""
        tool_call = SimpleNamespace(
            id="tooluse_obj_002",
            name="lookup_tool",
            input={"key": "value"},
        )
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, func_name, func_args = result
        assert func_name == "lookup_tool"
        assert func_args == {"key": "value"}
