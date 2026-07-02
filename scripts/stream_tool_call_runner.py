"""Real OpenAI Responses API runner for streamed tool-call deltas.

Fill in ``OPENAI_API_KEY`` below, then run from the repository root:

    uv run python scripts/stream_tool_call_runner.py
"""

from __future__ import annotations

import os

# ruff: noqa: T201
from crewai.llms.providers.openai.completion import OpenAICompletion
from dotenv import load_dotenv


load_dotenv()


MODEL = "gpt-4o-mini"


def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny and 72F."


def main() -> None:

    llm = OpenAICompletion(
        model=MODEL,
        api="responses",
        api_key=os.getenv("OPENAI_API_KEY"),
        stream=True,
        additional_params={
            "tool_choice": {"type": "function", "name": "get_weather"},
        },
    )

    tool_schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city to check.",
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }

    with llm.stream_events(
        "Call get_weather for Paris. Do not answer directly.",
        tools=[tool_schema],
        available_functions={"get_weather": get_weather},
    ) as stream:
        for frame in stream.llm:
            if frame.type != "llm_stream_chunk":
                continue

            tool_call = frame.data.get("tool_call")
            if not tool_call:
                print(f"text delta: {frame.content!r}")
                continue

            function = tool_call["function"]
            print(
                "tool delta: "
                f"chunk={frame.data['chunk']!r} "
                f"name={function['name']!r} "
                f"arguments={function['arguments']!r}"
            )

    print(f"final result: {stream.result!r}")


if __name__ == "__main__":
    main()
