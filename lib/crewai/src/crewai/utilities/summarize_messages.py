"""Structured context compaction for agent message histories."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import re
from typing import TYPE_CHECKING, Any

from crewai_core.printer import PRINTER

from crewai.utilities.agent_utils import (
    _content_parts_text,
    format_message_for_llm,
    is_inside_event_loop,
    message_content_text,
)
from crewai.utilities.i18n import I18N_DEFAULT
from crewai.utilities.types import LLMMessage


if TYPE_CHECKING:
    from crewai.llm import LLM
    from crewai.llms.base_llm import BaseLLM
    from crewai.utilities.token_counter_callback import TokenCalcHandler

_PART_PREFIX_TOKEN_RESERVE = 5


class SummarizeMessages:
    """Compact a message list so it fits the model context window.

    Preserves system messages, splits at message boundaries, formats with
    role labels, and writes one structured summary back onto ``messages``.
    Files attached to user messages are merged onto that summary.
    """

    def __init__(
        self,
        messages: list[LLMMessage],
        llm: LLM | BaseLLM,
        callbacks: list[TokenCalcHandler],
        verbose: bool = True,
    ) -> None:
        self.messages = messages
        self.llm = llm
        self.callbacks = callbacks
        self.verbose = verbose

    def summarize(self) -> None:
        """Replace non-system messages with a single summary, in place."""
        preserved_files = self._collect_attached_files()
        system_messages = [m for m in self.messages if m.get("role") == "system"]
        non_system_messages = [m for m in self.messages if m.get("role") != "system"]
        if not non_system_messages:
            return

        chunks = self._chunk_messages(
            non_system_messages, self.llm.get_context_window_size()
        )
        summaries = self._get_summaries_for_chunks(chunks)
        self._replace_history_with_summary(system_messages, summaries, preserved_files)

    def _get_summaries_for_chunks(self, chunks: list[list[LLMMessage]]) -> list[str]:
        total = len(chunks)
        if self.verbose and total <= 1:
            for index in range(1, total + 1):
                PRINTER.print(
                    content=f"Summarizing {index}/{total}...",
                    color="yellow",
                )
        if self.verbose and total > 1:
            PRINTER.print(
                content=f"Summarizing {total} chunks in parallel...",
                color="yellow",
            )
        return self._summarize_all(chunks)

    async def _summarize_one(self, chunk: list[LLMMessage]) -> str:
        """Summarize a single chunk."""
        summary = str(
            await self.llm.acall(
                self._build_summary_prompt(chunk), callbacks=self.callbacks
            )
        )
        match = re.search(r"<summary>(.*?)</summary>", summary, re.DOTALL)
        if match:
            return match.group(1).strip()
        return summary.strip()

    def _summarize_all(self, chunks: list[list[LLMMessage]]) -> list[str]:
        """Run one coroutine per chunk and return the summaries in order."""

        async def _gather() -> list[str]:
            coroutines = [self._summarize_one(chunk) for chunk in chunks]
            return list(await asyncio.gather(*coroutines))

        coro = _gather()
        if is_inside_event_loop():
            ctx = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(ctx.run, asyncio.run, coro).result()
        return asyncio.run(coro)

    def _build_summary_prompt(self, chunk: list[LLMMessage]) -> list[LLMMessage]:
        conversation = self._conversation_text(chunk)
        return [
            format_message_for_llm(
                I18N_DEFAULT.slice("summarizer_system_message"), role="system"
            ),
            format_message_for_llm(
                I18N_DEFAULT.slice("summarize_instruction").format(
                    conversation=conversation
                ),
            ),
        ]

    def _collect_attached_files(self) -> dict[str, Any]:
        preserved: dict[str, Any] = {}
        for msg in self.messages:
            if msg.get("role") == "user" and msg.get("files"):
                preserved.update(msg["files"])
        return preserved

    def _replace_history_with_summary(
        self,
        system_messages: list[LLMMessage],
        summaries: list[str],
        preserved_files: dict[str, Any],
    ) -> None:
        merged = "\n\n".join(summaries)
        summary_message = format_message_for_llm(
            I18N_DEFAULT.slice("summary").format(merged_summary=merged)
        )
        if preserved_files:
            summary_message["files"] = preserved_files

        self.messages.clear()
        self.messages.extend(system_messages)
        self.messages.append(summary_message)

    def _approx_tokens(self, text: str) -> int:
        """Estimate token count using roughly 1 token per 4 characters."""
        return len(text) // 4

    def _messages_ready_to_chunk(
        self, messages: list[LLMMessage], max_tokens: int
    ) -> list[LLMMessage]:
        """Drop system messages and split any entry that exceeds max_tokens."""
        ready: list[LLMMessage] = []
        for msg in messages:
            if msg.get("role") == "system":
                continue

            text = message_content_text(msg)
            if not text or self._approx_tokens(text) <= max_tokens:
                ready.append(msg)
                continue

            body_max_tokens = max(1, max_tokens - _PART_PREFIX_TOKEN_RESERVE)
            max_chars = max(1, body_max_tokens * 4)
            parts = [text[i : i + max_chars] for i in range(0, len(text), max_chars)]
            total_parts = len(parts)
            for index, part in enumerate(parts, start=1):
                ready.append(
                    {**msg, "content": f"[Part {index}/{total_parts}]\n{part}"}
                )
        return ready

    def _conversation_text(self, messages: list[LLMMessage]) -> str:
        """Format messages with role labels, skipping system messages."""
        lines: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                continue

            if role == "assistant":
                prefix = "[ASSISTANT]:"
            elif role == "tool":
                prefix = f"[TOOL_RESULT ({msg.get('name', 'unknown')})]:"
            else:
                prefix = "[USER]:"

            content = msg.get("content")
            if content is None:
                tool_calls = msg.get("tool_calls") or []
                names = []
                for tool_call in tool_calls:
                    func = tool_call.get("function", {})
                    names.append(
                        func.get("name", "unknown")
                        if isinstance(func, dict)
                        else "unknown"
                    )
                body = f"[Called tools: {', '.join(names)}]" if names else ""
            elif isinstance(content, list):
                body = _content_parts_text(content)
            else:
                body = str(content)

            lines.append(f"{prefix} {body}")
        return "\n\n".join(lines)

    def _chunk_messages(
        self, messages: list[LLMMessage], max_tokens: int
    ) -> list[list[LLMMessage]]:
        """Split messages into chunks that stay under max_tokens."""
        normalized = self._messages_ready_to_chunk(messages, max_tokens)
        if not normalized:
            return []

        chunks: list[list[LLMMessage]] = []
        current_chunk: list[LLMMessage] = []
        current_tokens = 0
        for msg in normalized:
            msg_tokens = self._approx_tokens(message_content_text(msg))
            if current_chunk and (current_tokens + msg_tokens) > max_tokens:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(msg)
            current_tokens += msg_tokens

        if current_chunk:
            chunks.append(current_chunk)
        return chunks
