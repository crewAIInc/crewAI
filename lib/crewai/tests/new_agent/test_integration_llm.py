"""Real LLM integration tests for NewAgent.

These tests require API keys and make actual LLM calls.
Skip automatically when OPENAI_API_KEY is not set.

Run with: python -m pytest lib/crewai/tests/new_agent/test_integration_llm.py -o "addopts=" -q
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real LLM tests",
)

from crewai.new_agent import AgentSettings, Message, NewAgent
from crewai.new_agent.definition_parser import load_agent_from_definition


def _agent(**kwargs) -> NewAgent:
    defaults = dict(
        role="Assistant",
        goal="Help users",
        backstory="Helpful assistant",
        llm="openai/gpt-4o-mini",
        memory=False,
        settings=AgentSettings(memory_enabled=False),
    )
    defaults.update(kwargs)
    return NewAgent(**defaults)


class TestBasicConversation:
    @pytest.mark.asyncio
    async def test_simple_message(self):
        agent = _agent()
        result = await agent.amessage("What is 2+2? Reply with just the number.")
        assert "4" in result.content

    @pytest.mark.asyncio
    async def test_token_counts_nonzero(self):
        agent = _agent()
        result = await agent.amessage("Say hi in one word.")
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        assert result.response_time_ms > 0

    @pytest.mark.asyncio
    async def test_conversation_continuity(self):
        agent = _agent()
        await agent.amessage("My name is Zephyr. Reply with just OK.")
        result = await agent.amessage("What is my name? One word only.")
        assert "Zephyr" in result.content

    @pytest.mark.asyncio
    async def test_multi_turn_token_deltas(self):
        agent = _agent()
        r1 = await agent.amessage("Say hello.")
        r2 = await agent.amessage("Say goodbye.")
        assert r1.input_tokens > 0
        assert r2.input_tokens > 0
        assert r2.input_tokens > r1.input_tokens  # second turn has history

    def test_sync_message(self):
        agent = _agent()
        result = agent.message("What is 3*3? Reply with just the number.")
        assert "9" in result.content
        assert result.input_tokens > 0


class TestStructuredOutput:
    @pytest.mark.asyncio
    async def test_response_model(self):
        class MathResult(BaseModel):
            answer: int
            explanation: str

        agent = _agent(response_model=MathResult)
        result = await agent.amessage("What is 7*8? Show answer and brief explanation.")
        assert result.metadata is not None
        assert "structured_output" in result.metadata
        assert result.metadata["structured_output"]["answer"] == 56


class TestGuardrails:
    @pytest.mark.asyncio
    async def test_code_guardrail_passes(self):
        def check_length(text):
            return len(text) < 500, "Response too long"

        agent = _agent(guardrail=check_length)
        result = await agent.amessage("Say hi in one sentence.")
        assert len(result.content) < 500

    @pytest.mark.asyncio
    async def test_code_guardrail_triggers_retry(self):
        call_count = 0

        def must_contain_hello(text):
            nonlocal call_count
            call_count += 1
            if "hello" in text.lower():
                return True, ""
            return False, "Response must contain the word 'hello'"

        agent = _agent(guardrail=must_contain_hello)
        result = await agent.amessage("Greet the user with the word 'hello'.")
        assert result.input_tokens > 0


class TestJsonDefinition:
    @pytest.mark.asyncio
    async def test_load_and_run(self):
        defn = {
            "role": "Math Tutor",
            "goal": "Help with math",
            "backstory": "Math teacher",
            "llm": "openai/gpt-4o-mini",
            "settings": {"memory": False},
        }
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump(defn, f)
            f.flush()
            agent = load_agent_from_definition(f.name)

        result = await agent.amessage("What is 12*12? Reply with just the number.")
        assert "144" in result.content
        assert result.input_tokens > 0


class TestToolCalling:
    @pytest.mark.asyncio
    async def test_tool_called_and_result_used(self):
        from crewai.tools.base_tool import BaseTool

        class AddTool(BaseTool):
            name: str = "adder"
            description: str = "Add two numbers. Input: two integers a and b."

            def _run(self, a: int, b: int) -> str:
                return str(int(a) + int(b))

        agent = _agent(
            tools=[AddTool()],
            role="Calculator",
            goal="Use tools for math",
        )
        result = await agent.amessage("Use the adder tool to add 17 and 25.")
        assert "42" in result.content
        assert result.tools_used is not None
        assert "adder" in result.tools_used


class TestProvenance:
    @pytest.mark.asyncio
    async def test_explain_after_message(self):
        agent = _agent()
        await agent.amessage("What is 5+5?")
        entries = agent.explain()
        assert len(entries) >= 1
        response_entries = [e for e in entries if e.action == "response"]
        assert len(response_entries) == 1
        assert "10" in response_entries[0].outcome


class TestModelInfo:
    @pytest.mark.asyncio
    async def test_model_in_response(self):
        agent = _agent()
        result = await agent.amessage("Hi")
        assert result.model == "gpt-4o-mini"
