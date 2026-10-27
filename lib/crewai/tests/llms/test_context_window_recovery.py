"""Hunt r5: context compaction recovery (#8277 regression probe).

The pre-refactor ``SummarizeMessages`` ladder re-chunked with tighter token
estimates when the provider rejected a summary request as too long. The
refactored ``llms/context_window.summarize_messages`` propagates the first
provider context error, so ``handle_context_length`` (documented to
"summarize or raise SystemExit") now kills the recovery path.
"""

from typing import Any

import pytest

from crewai.llms.base_llm import BaseLLM
from crewai.utilities.agent_utils import handle_context_length
from crewai.utilities.printer import Printer

# A provider whose real tokenizer is ~2 chars/token (token-dense text) while
# the estimator assumes 4 chars/token: the chunker packs twice what fits.
_REPLY_LIMIT_CHARS = 3550


class _TokenDenseLLM(BaseLLM):
    """Summary calls fail while the request is big, succeed once small."""

    def __init__(self) -> None:
        super().__init__(model="token-dense-test")

    def get_context_window_size(self) -> int:
        return 1000

    def get_input_tokens_count(self, messages: Any, tools=None, response_model=None) -> int:
        # Mirrors the chars/4 estimator used when a native counter is absent.
        total = 0
        for message in messages:
            content = message.get("content")
            total += len(content if isinstance(content, str) else str(content))
        return total // 4

    def call(self, messages: Any, **kwargs: Any) -> str:
        text = "".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        if len(text) > _REPLY_LIMIT_CHARS:
            raise RuntimeError(
                "This model's maximum context length is 1000 tokens, however you "
                "requested more tokens than the maximum context length"
            )
        return "<summary>short summary</summary>"

    async def acall(self, messages: Any, **kwargs: Any) -> str:
        return self.call(messages, **kwargs)


def _big_history() -> list[dict[str, Any]]:
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 500}
        for i in range(20)
    ]


def test_recovery_compacts_when_provider_rejects_the_first_chunk() -> None:
    """A recovery that can succeed on smaller chunks must not die on round 1."""
    llm = _TokenDenseLLM()
    messages = _big_history()

    handle_context_length(
        respect_context_window=True,
        printer=Printer(),
        messages=messages,
        llm=llm,
        callbacks=[],
        verbose=False,
    )

    assert len(messages) == 1  # history compacted to a single summary message
    assert messages[-1]["role"] == "user"
    assert "short summary" in messages[-1]["content"]
