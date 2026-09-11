from pydantic import BaseModel

from crewai.llms.providers.utils.common import safe_tool_conversion
from crewai.tools.base_tool import BaseTool


class EchoArgs(BaseModel):
    text: str


class EchoTool(BaseTool):
    name: str = "Echo Tool"
    description: str = "Echo the provided text."
    args_schema: type[BaseModel] = EchoArgs

    def _run(self, text: str) -> str:
        return text


def test_safe_tool_conversion_accepts_base_tool():
    tool = EchoTool()

    name, description, parameters = safe_tool_conversion(
        tool,
        "OpenAI",
    )

    assert name == "echo_tool"
    assert description == "Echo the provided text."
    assert "text" in parameters["properties"]

def test_safe_tool_conversion_accepts_dictionary_tool():
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                    }
                },
                "required": ["city"],
            },
        },
    }

    name, description, parameters = safe_tool_conversion(
        tool,
        "OpenAI",
    )

    assert name == "get_weather"
    assert description == "Get the weather for a city."
    assert "city" in parameters["properties"]