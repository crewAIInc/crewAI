"""CrewAI Flow that streams model-produced OpenUI Lang over AG-UI."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ag_ui_crewai.sdk import CopilotKitState, copilotkit_stream
from crewai.flow.flow import Flow, start
from litellm import acompletion

DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "frontend" / "generated" / "system-prompt.txt"
)


def load_system_prompt() -> str:
    """Load the prompt generated from frontend/src/library.ts."""
    prompt_path = Path(os.getenv("OPENUI_SYSTEM_PROMPT_PATH", str(DEFAULT_PROMPT_PATH)))
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RuntimeError(
            "OpenUI system prompt is missing. Run `npm run generate:prompt` "
            "from the example's frontend directory before starting the server."
        ) from error


def message_for_llm(message: Any) -> Any:
    """Convert AG-UI messages to provider-safe chat-completion messages."""
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(exclude_none=True)
    elif isinstance(message, Mapping):
        payload = dict(message)
    else:
        return message

    # AG-UI allows an empty tool_calls list. OpenAI requires the field to be
    # absent unless at least one tool call exists.
    if payload.get("role") == "assistant" and payload.get("tool_calls") == []:
        payload.pop("tool_calls")
    return payload


def build_messages(system_prompt: str, messages: Sequence[Any]) -> list[Any]:
    """Prepend the generated prompt without rewriting interaction context."""
    return [
        {"role": "system", "content": system_prompt},
        *(message_for_llm(message) for message in messages),
    ]


class OpenUIFlow(Flow[CopilotKitState]):
    """Handle every chat, follow-up, and form action as a CrewAI Flow turn."""

    @start()
    async def chat(self) -> None:
        """Stream one OpenUI response and persist its assistant message."""
        response = await copilotkit_stream(
            await acompletion(
                model=os.getenv("OPENUI_MODEL", "openai/gpt-4.1-mini"),
                messages=build_messages(load_system_prompt(), self.state.messages),
                stream=True,
            )
        )

        self.state.messages.append(response.choices[0].message)
