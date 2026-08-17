from typing import Any
from unittest.mock import MagicMock, patch

from crewai import LLM
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.llms.providers.utils.common import (
    extract_tool_info,
    safe_tool_conversion,
)
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pytest


class _EchoInput(BaseModel):
    text: str = Field(..., description="Text to echo")


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Echo the provided text."
    args_schema: type[BaseModel] = _EchoInput

    def _run(self, text: str) -> str:
        return text


DICT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def test_extract_tool_info_rejects_plain_objects() -> None:
    with pytest.raises(ValueError, match="Tool must be a dictionary"):
        extract_tool_info(object())


def test_extract_tool_info_accepts_base_tool() -> None:
    name, description, parameters = extract_tool_info(EchoTool())

    assert name == "echo"
    assert description == "Echo the provided text."
    assert parameters["type"] == "object"
    assert "text" in parameters["properties"]
    assert parameters["properties"]["text"]["type"] == "string"
    assert "text" in parameters["required"]


def test_extract_tool_info_accepts_openai_dict() -> None:
    name, description, parameters = extract_tool_info(DICT_TOOL)

    assert name == "get_weather"
    assert description == "Get the weather for a city."
    assert parameters["type"] == "object"
    assert "city" in parameters["properties"]
    assert parameters["required"] == ["city"]


def test_safe_tool_conversion_accepts_base_tool() -> None:
    name, description, parameters = safe_tool_conversion(EchoTool(), "OpenAI")

    assert name == "echo"
    assert description == "Echo the provided text."
    assert parameters["type"] == "object"
    assert "text" in parameters["properties"]
    assert parameters["properties"]["text"]["type"] == "string"
    assert "text" in parameters["required"]


def test_openai_converts_base_tool_for_interference() -> None:
    llm = OpenAICompletion(model="gpt-4o-mini", api_key="test-key")
    converted = llm._convert_tools_for_interference([EchoTool()])  # type: ignore[list-item]

    assert converted[0]["type"] == "function"
    assert converted[0]["function"]["name"] == "echo"
    assert "text" in converted[0]["function"]["parameters"]["properties"]


def test_llm_call_sends_converted_base_tool_schema() -> None:
    llm = LLM(model="gpt-4o-mini", api_key="test-key")

    with patch.object(llm._client.chat.completions, "create") as mock_create:
        mock_create.return_value = MagicMock(
            choices=[
                MagicMock(message=MagicMock(content="ok", tool_calls=None))
            ],
            usage=MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

        result = llm.call(
            messages=[{"role": "user", "content": "Who are you?"}],
            tools=[EchoTool()],
        )

    assert result == "ok"
    mock_create.assert_called_once()
    sent_tools = mock_create.call_args.kwargs["tools"]
    function = sent_tools[0]["function"]
    parameters = function["parameters"]

    assert sent_tools[0]["type"] == "function"
    assert function["name"] == "echo"
    assert function["description"] == "Echo the provided text."
    assert parameters["type"] == "object"
    assert "text" in parameters["properties"]
    assert parameters["properties"]["text"]["type"] == "string"
    assert "text" in parameters["required"]


def test_anthropic_converts_base_tool_before_dict_checks() -> None:
    pytest.importorskip("anthropic")
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    llm = AnthropicCompletion(model="claude-sonnet-4-5", api_key="test-key")
    converted = llm._convert_tools_for_interference([EchoTool()])  # type: ignore[list-item]

    tool: dict[str, Any] = converted[0]
    assert tool["name"] == "echo"
    assert tool["description"] == "Echo the provided text."
    assert tool["input_schema"]["type"] == "object"
    assert "text" in tool["input_schema"]["properties"]
