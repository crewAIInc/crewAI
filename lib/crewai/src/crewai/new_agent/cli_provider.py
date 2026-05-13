"""Terminal-based conversational provider for NewAgent."""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import threading
from typing import TYPE_CHECKING, Any

from crewai.new_agent.models import AgentStatus, Message, ProvenanceEntry

if TYPE_CHECKING:
    from crewai.new_agent.provider import SQLiteConversationStorage


# ── Spinner frames ───────────────────────────────────────────

_BRAILLE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# ── Formatting helpers ───────────────────────────────────────


def format_tokens(n: int) -> str:
    """Format a token count compactly.

    Examples:
        0     → "0"
        999   → "999"
        1000  → "1.0k"
        1234  → "1.2k"
        12345 → "12.3k"
        1234567 → "1.2M"
    """
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        value = n / 1000
        return f"{value:.1f}k"
    value = n / 1_000_000
    return f"{value:.1f}M"


def format_elapsed(ms: int) -> str:
    """Format elapsed milliseconds as a human-readable duration.

    Examples:
        12000   → "12s"
        72000   → "1m 12s"
        3723000 → "1h 2m"
    """
    total_seconds = ms // 1000
    if total_seconds < 60:
        return f"{total_seconds}s"
    if total_seconds < 3600:
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}m {seconds}s"
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours}h {minutes}m"


def format_status_line(status: AgentStatus, spinner_frame: str = "⠋") -> str:
    """Build the status line shown during agent work.

    Format:
        ⠋ Searching the web… (12s · ↓ 3.4k tokens · ↑ 1.2k tokens)
    """
    detail = status.detail or status.state
    parts: list[str] = []
    if status.elapsed_ms:
        parts.append(format_elapsed(status.elapsed_ms))
    if status.input_tokens:
        parts.append(f"↓ {format_tokens(status.input_tokens)} tokens")
    if status.output_tokens:
        parts.append(f"↑ {format_tokens(status.output_tokens)} tokens")
    suffix = f" ({' · '.join(parts)})" if parts else ""
    return f"{spinner_frame} {detail}…{suffix}"


# ── Spinner helper ───────────────────────────────────────────


class _Spinner:
    """Simple terminal spinner that overwrites the current line."""

    def __init__(self) -> None:
        self._running = False
        self._thread: threading.Thread | None = None
        self._status: AgentStatus | None = None
        self._lock = threading.Lock()

    def update(self, status: AgentStatus) -> None:
        with self._lock:
            self._status = status

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        # Clear the spinner line
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    def _spin(self) -> None:
        frames = _BRAILLE_FRAMES
        idx = 0
        while self._running:
            with self._lock:
                status = self._status
            if status is not None:
                frame = frames[idx % len(frames)]
                line = format_status_line(status, spinner_frame=frame)
                sys.stderr.write(f"\r\033[K{line}")
                sys.stderr.flush()
            idx += 1
            try:
                # ~80ms per frame ≈ 12.5 fps
                threading.Event().wait(timeout=0.08)
            except Exception:
                break


# ── History persistence ──────────────────────────────────────


def _storage_path(agent_name: str) -> Path:
    """Return the path to the agent's SQLite conversation database."""
    return Path.cwd() / ".crewai" / "conversations" / f"{agent_name}.db"


def _get_storage(agent_name: str) -> SQLiteConversationStorage:
    from crewai.new_agent.provider import SQLiteConversationStorage
    return SQLiteConversationStorage(_storage_path(agent_name))


# ── CLIProvider ──────────────────────────────────────────────


class CLIProvider:
    """Terminal-based conversational provider for NewAgent.

    Uses stdin/stdout for user interaction and displays live status
    updates with an animated spinner on stderr.  Conversation history
    is persisted via SQLiteConversationStorage (WAL mode).
    """

    def __init__(self, agent_name: str = "agent", storage: Any = None) -> None:
        self.agent_name = agent_name
        self._storage = storage or _get_storage(agent_name)
        self._spinner = _Spinner()

    # ── ConversationalProvider protocol ──────────────────────

    async def send_message(self, message: Message) -> None:
        """Print the agent's message to stdout."""
        # Stop spinner before printing output
        self._spinner.stop()

        prefix = ""
        if message.role == "agent":
            prefix = f"\n{message.sender or 'Agent'}: " if message.sender else "\nAgent: "
        elif message.role == "system":
            prefix = "\n[system] "

        sys.stdout.write(f"{prefix}{message.content}\n")
        sys.stdout.flush()

    async def receive_message(self) -> Message:
        """Read user input from stdin."""
        # Stop spinner while waiting for input
        self._spinner.stop()

        try:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(None, self._read_input)
        except EOFError as err:
            raise KeyboardInterrupt("End of input") from err

        return Message(role="user", content=text)

    async def send_status(self, status: AgentStatus) -> None:
        """Show a spinner with status details on stderr."""
        self._spinner.update(status)
        self._spinner.start()

    def get_history(self) -> list[Message]:
        return self._storage.load_messages()

    def save_history(self, messages: list[Message]) -> None:
        self._storage.save_messages(messages)

    def reset_history(self) -> None:
        self._storage.clear_messages()

    def save_provenance(self, entries: list[ProvenanceEntry]) -> None:
        self._storage.save_provenance(entries)

    def load_provenance(self) -> list[ProvenanceEntry]:
        return self._storage.load_provenance()

    def get_scope(self) -> dict[str, str]:
        return {}

    # ── Internal helpers ─────────────────────────────────────

    @staticmethod
    def _read_input() -> str:
        """Blocking stdin read (called from executor)."""
        return input("\nYou: ")
