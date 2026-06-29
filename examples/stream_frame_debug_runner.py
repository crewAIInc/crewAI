"""Minimal direct LLM streaming runner.

Run from the repo root:

    uv run python examples/stream_frame_debug_runner.py
"""

from __future__ import annotations

# ruff: noqa: T201
import os

from crewai import LLM


llm = LLM(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
messages = [
    {
        "role": "user",
        "content": "Explain CrewAI streaming in two short sentences.",
    }
]

stream = llm.stream_events(messages=messages)

print("--- chunks ---")
with stream:
    for chunk in stream:
        print(chunk.content, end="", flush=True)

print("\n\n--- result ---")
print(stream.result)
