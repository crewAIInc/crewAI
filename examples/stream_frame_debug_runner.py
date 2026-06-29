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

chunks = llm.stream_call(messages=messages)

print("--- chunks ---")
for chunk in chunks:
    print(chunk.content, end="", flush=True)

# print("\n\n--- result ---")
# print(chunks.result)
