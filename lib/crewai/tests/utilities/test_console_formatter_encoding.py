import io
import sys

from crewai.events.utils.console_formatter import ConsoleFormatter


def _gbk_stdout() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="gbk", errors="strict")


def _drain(stream: io.TextIOWrapper) -> str:
    stream.flush()
    return stream.buffer.getvalue().decode(stream.encoding or "utf-8")


class TestConsoleFormatterEncoding:
    """Panels must render even when stdout cannot encode emoji (Windows GBK)."""

    def test_emoji_panels_render_on_non_utf8_stdout(self, monkeypatch):
        """Redirected output on a CJK-locale Windows console encodes as GBK and
        cannot hold the emoji in panel titles; the handler must still print the
        panel (with unencodable characters replaced) instead of raising."""
        fake = _gbk_stdout()
        monkeypatch.setattr(sys, "stdout", fake)

        formatter = ConsoleFormatter()
        formatter.handle_flow_started("my_flow", "abc-123")
        formatter.handle_method_status("step", "completed")

        output = _drain(fake)
        assert "Flow Started" in output
        assert "Completed" in output

    def test_ascii_panel_unchanged_on_non_utf8_stdout(self, monkeypatch):
        """ASCII-only panels print the same text on a GBK stream (guard)."""
        fake = _gbk_stdout()
        monkeypatch.setattr(sys, "stdout", fake)

        formatter = ConsoleFormatter(verbose=True)
        formatter.handle_crew_test_completed("my_crew")

        output = _drain(fake)
        assert "Test Execution Completed" in output
        assert "my_crew" in output

    def test_emoji_kept_intact_on_utf8_stdout(self, monkeypatch):
        """Streams that can encode emoji keep them — the fallback only degrades
        output when the stream actually cannot encode it (guard)."""
        fake = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
        monkeypatch.setattr(sys, "stdout", fake)

        formatter = ConsoleFormatter()
        formatter.handle_flow_started("my_flow", "abc-123")

        output = _drain(fake)
        assert "🌊 Flow Started" in output
