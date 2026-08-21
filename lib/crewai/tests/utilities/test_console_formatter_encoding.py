import io

from rich.text import Text

from crewai.events.utils.console_formatter import ConsoleFormatter


class TestConsoleFormatterEncoding:
    """Regression tests: emoji panel titles must not crash on non-UTF-8 streams (e.g. Windows cp1252 console)."""

    def test_reconfigures_stream_errors_to_replace(self):
        formatter = ConsoleFormatter()

        assert formatter.console.file.errors == "replace"

    def test_emoji_panel_does_not_raise_on_cp1252_stream(self):
        cp1252_stream = io.TextIOWrapper(
            io.BytesIO(), encoding="cp1252", write_through=True
        )
        formatter = ConsoleFormatter()
        formatter.console.file = cp1252_stream
        if hasattr(cp1252_stream, "reconfigure"):
            cp1252_stream.reconfigure(errors="replace")

        formatter.print_panel(Text("x"), "🌊 Flow Started", "blue", is_flow=True)
