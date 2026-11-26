"""Runnable example for Flow.ask() with a scripted custom input provider.

Documented in docs/edge/en/concepts/flows.mdx ("Custom input provider for Flow.ask()").
No stdin, LLM calls, or credentials required.

Run:
    cd lib/crewai && uv run python examples/flow_custom_input_provider.py
"""

from __future__ import annotations

from typing import Any

from crewai.flow import Flow, start
from crewai.flow.input_provider import InputResponse


class ScriptedProvider:
    """Returns pre-scripted replies; records each (message, metadata) pair."""

    def __init__(self, replies: list[str | InputResponse | None]) -> None:
        self.replies = list(replies)
        self.seen: list[tuple[str, dict[str, Any] | None]] = []

    def request_input(
        self,
        message: str,
        flow: Flow[Any],
        metadata: dict[str, Any] | None = None,
    ) -> str | InputResponse | None:
        self.seen.append((message, metadata))
        if not self.replies:
            return None
        return self.replies.pop(0)


class ResearchWizardFlow(Flow):
    input_provider = ScriptedProvider(
        [
            InputResponse(
                text="vector databases",
                metadata={"answered_by": "script", "thread_id": "t1"},
            )
        ]
    )

    @start()
    def gather(self) -> dict[str, str | None]:
        topic = self.ask(
            "What should we research?",
            metadata={"channel": "#research"},
        )
        if topic is None:
            return {"status": "cancelled", "topic": None, "depth": None}

        depth = self.ask(
            "How deep should we go?",
            metadata={"channel": "#research"},
        )
        if depth is None:
            return {"status": "partial", "topic": topic, "depth": None}

        return {"status": "ok", "topic": topic, "depth": depth}


def main() -> None:
    flow = ResearchWizardFlow()
    result = flow.kickoff()

    print("result:", result)
    print("provider saw:", flow.input_provider.seen)
    first = flow.input_history[0]
    print("response_metadata:", first["response_metadata"])


if __name__ == "__main__":
    main()
