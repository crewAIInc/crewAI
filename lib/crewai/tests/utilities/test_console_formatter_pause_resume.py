import io
import re
import time
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest
from rich.console import Console
from rich.live import Live

from crewai.events.types.llm_events import LLMCallType
from crewai.events.utils.console_formatter import ConsoleFormatter


class _TTYStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class TestConsoleFormatterPauseResume:
    """Test ConsoleFormatter pause/resume functionality for HITL features."""

    def test_pause_stops_active_streaming_session(self):
        """Test pausing stops an active streaming Live session."""
        formatter = ConsoleFormatter()

        mock_live = MagicMock(spec=Live)
        formatter._streaming_live = mock_live

        formatter.pause_live_updates()

        mock_live.stop.assert_called_once()
        assert formatter._streaming_live is None

    def test_pause_is_safe_when_no_session(self):
        """Test pausing when no streaming session exists doesn't error."""
        formatter = ConsoleFormatter()
        formatter._streaming_live = None

        formatter.pause_live_updates()

        assert formatter._streaming_live is None

    def test_multiple_pauses_are_safe(self):
        """Test calling pause multiple times is safe."""
        formatter = ConsoleFormatter()

        mock_live = MagicMock(spec=Live)
        formatter._streaming_live = mock_live

        formatter.pause_live_updates()
        mock_live.stop.assert_called_once()
        assert formatter._streaming_live is None

        formatter.pause_live_updates()

    def test_resume_is_safe(self):
        """Test resume method exists and doesn't error."""
        formatter = ConsoleFormatter()

        formatter.resume_live_updates()

    def test_flow_method_status_ignores_formatter_verbose(self):
        formatter = ConsoleFormatter(verbose=False)

        with patch.object(formatter, "print_panel") as mock_print_panel:
            formatter.handle_method_status("categorize_tickets")

        mock_print_panel.assert_called_once()
        _, kwargs = mock_print_panel.call_args
        assert kwargs["is_flow"] is True

    def test_streaming_after_pause_resume_creates_new_session(self):
        """Test that streaming after pause/resume creates new Live session."""
        formatter = ConsoleFormatter()
        formatter.verbose = True
        formatter.console = Console(file=_TTYStringIO())

        # Simulate having an active session
        mock_live = MagicMock(spec=Live)
        formatter._streaming_live = mock_live

        # Pause stops the session
        formatter.pause_live_updates()
        assert formatter._streaming_live is None

        # Resume (no-op, sessions created on demand)
        formatter.resume_live_updates()

        # After resume, streaming should be able to start a new session
        with patch("crewai.events.utils.console_formatter.Live") as mock_live_class:
            mock_live_instance = MagicMock()
            mock_live_class.return_value = mock_live_instance

            # Simulate streaming chunk (this creates a new Live session)
            formatter.handle_llm_stream_chunk("test chunk", call_type=None)

            mock_live_class.assert_called_once()
            mock_live_instance.start.assert_called_once()
            assert formatter._streaming_live == mock_live_instance

    def test_pause_resume_cycle_with_streaming(self):
        """Test full pause/resume cycle during streaming."""
        formatter = ConsoleFormatter()
        formatter.verbose = True
        formatter.console = Console(file=_TTYStringIO())

        with patch("crewai.events.utils.console_formatter.Live") as mock_live_class:
            mock_live_instance = MagicMock()
            mock_live_class.return_value = mock_live_instance

            formatter.handle_llm_stream_chunk("chunk 1", call_type=None)
            assert formatter._streaming_live == mock_live_instance

            # Pause should stop the session
            formatter.pause_live_updates()
            mock_live_instance.stop.assert_called_once()
            assert formatter._streaming_live is None

            # Resume (no-op)
            formatter.resume_live_updates()

            mock_live_instance_2 = MagicMock()
            mock_live_class.return_value = mock_live_instance_2

            # Streaming again creates new session
            formatter.handle_llm_stream_chunk("chunk 2", call_type=None)
            assert formatter._streaming_live == mock_live_instance_2


FINAL_ANSWER_TITLE = "Agent Final Answer"


@pytest.fixture
def clean_terminal_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in ("FORCE_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE", "NO_COLOR"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    return monkeypatch


def _streaming_formatter(file: io.StringIO, **console_kwargs: Any) -> ConsoleFormatter:
    formatter = ConsoleFormatter(verbose=True)
    formatter.console = Console(file=file, width=80, **console_kwargs)
    return formatter


def _stream(formatter: ConsoleFormatter, chunks: int, call_type: Any) -> None:
    text = ""
    for i in range(chunks):
        text += f"word{i} " + ("\n" if i % 8 == 7 else "")
        formatter.handle_llm_stream_chunk(text, call_type=call_type)


class _WriteOnlyFile:
    """A stdout replacement (tee, log shim) that has no isatty."""

    def __init__(self, target: io.StringIO) -> None:
        self._target = target

    def write(self, text: str) -> int:
        return self._target.write(text)

    def flush(self) -> None:
        self._target.flush()


class TestStreamingLiveOffTerminal:
    """Forced colour on a redirected stdout must not append a panel per repaint."""

    @pytest.mark.parametrize(
        ("env_name", "env_value"),
        [("FORCE_COLOR", "1"), ("FORCE_COLOR", "0"), ("TTY_COMPATIBLE", "1")],
    )
    def test_forced_terminal_on_file_renders_final_panel_once(
        self, clean_terminal_env: pytest.MonkeyPatch, env_name: str, env_value: str
    ) -> None:
        clean_terminal_env.setenv(env_name, env_value)
        buf = io.StringIO()
        formatter = _streaming_formatter(buf)

        _stream(formatter, 200, LLMCallType.LLM_CALL)
        formatter.handle_llm_stream_completed()

        output = buf.getvalue()
        assert output.count(FINAL_ANSWER_TITLE) == 1
        assert "word199" in output
        assert re.search(r"\x1b\[[0-9;]*m", output), "forced colour must be kept"
        assert len(output) < 10_000

    def test_forced_terminal_on_file_renders_tool_arguments_once(
        self, clean_terminal_env: pytest.MonkeyPatch
    ) -> None:
        clean_terminal_env.setenv("FORCE_COLOR", "1")
        buf = io.StringIO()
        formatter = _streaming_formatter(buf)

        _stream(formatter, 100, LLMCallType.TOOL_CALL)
        formatter.handle_llm_stream_completed()

        output = buf.getvalue()
        assert output.count("Tool Arguments") == 1
        assert "word99" in output

    def test_forced_terminal_on_file_writes_nothing_while_stream_stalls(
        self, clean_terminal_env: pytest.MonkeyPatch
    ) -> None:
        clean_terminal_env.setenv("FORCE_COLOR", "1")
        buf = io.StringIO()
        formatter = _streaming_formatter(buf)

        try:
            formatter.handle_llm_stream_chunk("partial", call_type=LLMCallType.LLM_CALL)
            after_first_chunk = buf.getvalue()
            time.sleep(0.35)
            assert buf.getvalue() == after_first_chunk
            assert FINAL_ANSWER_TITLE not in after_first_chunk
        finally:
            formatter.handle_llm_stream_completed()

        assert buf.getvalue().count(FINAL_ANSWER_TITLE) == 1

    def test_pause_on_file_flushes_panel_once(
        self, clean_terminal_env: pytest.MonkeyPatch
    ) -> None:
        clean_terminal_env.setenv("FORCE_COLOR", "1")
        buf = io.StringIO()
        formatter = _streaming_formatter(buf)

        _stream(formatter, 50, LLMCallType.LLM_CALL)
        formatter.pause_live_updates()
        after_pause = buf.getvalue()
        formatter.handle_llm_stream_completed()

        assert after_pause.count(FINAL_ANSWER_TITLE) == 1
        assert "word49" in after_pause
        assert buf.getvalue() == after_pause

    def test_unforced_file_output_is_one_plain_panel(
        self, clean_terminal_env: pytest.MonkeyPatch
    ) -> None:
        buf = io.StringIO()
        formatter = _streaming_formatter(buf)

        _stream(formatter, 100, LLMCallType.LLM_CALL)
        formatter.handle_llm_stream_completed()

        output = buf.getvalue()
        assert output.count(FINAL_ANSWER_TITLE) == 1
        assert "word99" in output
        assert "\x1b[" not in output

    def test_chunk_on_closed_stream_does_not_raise(
        self, clean_terminal_env: pytest.MonkeyPatch
    ) -> None:
        buf = io.StringIO()
        formatter = _streaming_formatter(buf)
        buf.close()

        formatter.handle_llm_stream_chunk("partial", call_type=LLMCallType.LLM_CALL)

    def test_file_without_isatty_renders_final_panel_once(
        self, clean_terminal_env: pytest.MonkeyPatch
    ) -> None:
        clean_terminal_env.setenv("FORCE_COLOR", "1")
        buf = io.StringIO()
        formatter = _streaming_formatter(_WriteOnlyFile(buf))

        _stream(formatter, 50, LLMCallType.LLM_CALL)
        formatter.handle_llm_stream_completed()

        output = buf.getvalue()
        assert output.count(FINAL_ANSWER_TITLE) == 1
        assert "word49" in output

    @pytest.mark.parametrize("via_console", [True, False])
    def test_prints_during_stream_do_not_repeat_panel(
        self,
        clean_terminal_env: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        via_console: bool,
    ) -> None:
        clean_terminal_env.setenv("FORCE_COLOR", "1")
        buf = io.StringIO()
        formatter = _streaming_formatter(buf)

        text = ""
        for i in range(30):
            text += f"word{i} "
            formatter.handle_llm_stream_chunk(text, call_type=LLMCallType.LLM_CALL)
            if via_console:
                formatter.console.print(f"side note {i}")
            else:
                print(f"side note {i}")
        formatter.handle_llm_stream_completed()

        output = buf.getvalue() + capsys.readouterr().out
        assert output.count(FINAL_ANSWER_TITLE) == 1
        assert output.count("side note") == 30
        assert "word29" in output


class TestStreamingLiveOnTerminal:
    """A real terminal and Jupyter keep the animated panel."""

    def test_tty_repaints_panel_per_chunk(
        self, clean_terminal_env: pytest.MonkeyPatch
    ) -> None:
        buf = _TTYStringIO()
        formatter = _streaming_formatter(buf)

        try:
            _stream(formatter, 20, LLMCallType.LLM_CALL)
        finally:
            formatter.handle_llm_stream_completed()

        output = buf.getvalue()
        assert output.count(FINAL_ANSWER_TITLE) >= 20
        assert "\x1b[2K" in output, "cursor-erase repaints expected on a terminal"

    def test_jupyter_keeps_live_panel(
        self, clean_terminal_env: pytest.MonkeyPatch
    ) -> None:
        formatter = _streaming_formatter(io.StringIO(), force_jupyter=True)

        with patch("crewai.events.utils.console_formatter.Live") as mock_live_class:
            formatter.handle_llm_stream_chunk("partial", call_type=LLMCallType.LLM_CALL)
            formatter.handle_llm_stream_chunk(
                "partial 2", call_type=LLMCallType.LLM_CALL
            )

        mock_live_class.return_value.start.assert_called_once_with()
        mock_live_class.return_value.update.assert_called_once_with(ANY, refresh=True)
