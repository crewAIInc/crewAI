"""Bounded output collection using the sprites-py 0.6 WebSocket transport.

The SDK's high-level run and streaming APIs both buffer all output. Intercept
output before its collector, while retaining its connection and exit handling.
Keep this adapter covered by tests against the installed, version-bounded SDK.
"""

from __future__ import annotations

import codecs
from contextlib import suppress
from typing import TYPE_CHECKING

from sprites.websocket import StreamID, WSCommand


if TYPE_CHECKING:
    from sprites.exec import Cmd


class _BoundedOutput:
    """Retain a UTF-8 prefix without accumulating discarded output."""

    def __init__(self, limit: int) -> None:
        """Allocate an incremental decoder and a character-bounded collector."""
        self.limit = limit
        self.length = 0
        self.parts: list[str] = []
        self.truncated = False
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def write(self, data: bytes, *, final: bool = False) -> None:
        """Decode only enough bytes to retain the prefix and detect overflow."""
        if self.truncated:
            return
        remaining = self.limit - self.length
        # UTF-8 needs at most four bytes per character. One extra character
        # distinguishes exact-length output from truncated output.
        byte_limit = 4 * (remaining + 1)
        decoded = self.decoder.decode(data[:byte_limit], final=final)
        prefix = decoded[:remaining]
        if prefix:
            self.parts.append(prefix)
            self.length += len(prefix)
        self.truncated = len(decoded) > remaining or len(data) > byte_limit

    def text(self) -> str:
        """Flush any incomplete final UTF-8 sequence and return the prefix."""
        self.write(b"", final=True)
        return "".join(self.parts)


class BoundedWSCommand(WSCommand):
    """Drain a non-TTY command while retaining at most limit chars per stream."""

    def __init__(self, cmd: Cmd, limit: int) -> None:
        """Attach bounded collectors before the SDK starts receiving frames."""
        super().__init__(cmd)
        self.stdout = _BoundedOutput(limit)
        self.stderr = _BoundedOutput(limit)

    async def _handle_message(self, message: str | bytes) -> None:
        """Intercept output; delegate control and exit frames to the SDK."""
        if self.cmd.tty:
            raise RuntimeError("SpritesExecTool requires non-TTY command output.")
        if isinstance(message, bytes) and message:
            if message[0] == StreamID.STDOUT:
                self.stdout.write(message[1:])
                return
            if message[0] == StreamID.STDERR:
                self.stderr.write(message[1:])
                return
        await super()._handle_message(message)

    async def execute(self) -> int:
        """Run and close the local connection on success, error, or cancellation."""
        try:
            await self.start()
            return await self.wait()
        finally:
            # A close failure must not hide the exit status or original error.
            with suppress(Exception):
                await self.close()
