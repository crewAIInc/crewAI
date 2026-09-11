"""Keep unauthenticated execution spans local until explicit upload consent."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import logging
import os
from threading import Lock

from opentelemetry.exporter.otlp.proto.common.trace_encoder import encode_spans
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExportResult,
    SpanExporter,
)

from crewai.events.listeners.tracing.utils import prompt_user_for_trace_viewing
from crewai.telemetry.tracing.grants import (
    GrantSpanExporter,
    TraceGrantClient,
    TraceGrantError,
)
from crewai.telemetry.tracing.session import TraceSession


logger = logging.getLogger(__name__)


def _positive_limit(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
        if value > 0:
            return value
    except ValueError:
        pass
    logger.warning("Invalid %s; using default limit %d", name, default)
    return default


class EphemeralSpanBuffer(SpanExporter):
    """Retain recent spans within count and encoded OTLP byte limits.

    Defaults are 1,000 spans and 8 MiB, configurable with
    ``CREWAI_EPHEMERAL_TRACE_MAX_SPANS`` and ``CREWAI_EPHEMERAL_TRACE_MAX_BYTES``.
    Invalid or nonpositive limits warn and fall back to these defaults.
    Overflow evicts the oldest spans; an individually oversized span is dropped.
    Nothing leaves the process through ``export``.
    """

    def __init__(self) -> None:
        self._max_spans = _positive_limit("CREWAI_EPHEMERAL_TRACE_MAX_SPANS", 1000)
        self._max_bytes = _positive_limit("CREWAI_EPHEMERAL_TRACE_MAX_BYTES", 8388608)
        self._spans: deque[tuple[ReadableSpan, int]] = deque()
        self._size = 0
        self._dropped = 0
        self._closed = False
        self._lock = Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._lock:
            if self._closed:
                return SpanExportResult.FAILURE
            for span in spans:
                size = encode_spans([span]).ByteSize()
                if size > self._max_bytes:
                    self._dropped += 1
                    continue
                while self._spans and (
                    len(self._spans) >= self._max_spans
                    or self._size + size > self._max_bytes
                ):
                    _, removed_size = self._spans.popleft()
                    self._size -= removed_size
                    self._dropped += 1
                self._spans.append((span, size))
                self._size += size
        return SpanExportResult.SUCCESS

    def share(self, execution_uuid: str) -> None:
        """Ask once, then obtain a grant and use the shared OTLP exporter."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            buffered = [span for span, _ in self._spans]
            self._spans.clear()
            self._size = 0
        try:
            if self._dropped:
                logger.warning(
                    "Ephemeral trace buffer dropped %d spans at its configured limits",
                    self._dropped,
                )
            if not buffered:
                return
            if not prompt_user_for_trace_viewing(sharing=True):
                logger.info("Ephemeral trace discarded without uploading")
                return
            client = TraceGrantClient(None)
            grant = client.create(execution_uuid)
            exporter = GrantSpanExporter(client, grant)
            try:
                if exporter.export(buffered) != SpanExportResult.SUCCESS:
                    logger.warning("Ephemeral trace export failed; buffer discarded")
            finally:
                exporter.shutdown()
        except TraceGrantError as error:
            logger.warning(
                "Ephemeral trace grant failed (HTTP %s); buffer discarded",
                error.status_code,
            )
        except Exception as error:
            # Do not log prompts, credentials, or server response bodies.
            logger.warning("Ephemeral trace sharing failed (%s)", type(error).__name__)
        finally:
            buffered.clear()
            self.shutdown()

    def shutdown(self) -> None:
        with self._lock:
            self._closed = True
            self._spans.clear()
            self._size = 0


@contextmanager
def ephemeral_tracing(execution_uuid: str) -> Iterator[TraceSession]:
    """Own buffering and consent; activation can change between deferred turns."""
    buffer = EphemeralSpanBuffer()
    session = TraceSession(execution_uuid, processors=[SimpleSpanProcessor(buffer)])
    try:
        yield session
        session.finish_spans()
        if session.flush():
            buffer.share(execution_uuid)
    finally:
        buffer.shutdown()
        session.shutdown()
