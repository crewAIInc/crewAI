from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from crewai.utilities.printer import ColoredText, Printer, PrinterColor


class Logger(BaseModel):
    verbose: bool = Field(
        default=False,
        description="Enables verbose logging with timestamps",
    )
    default_color: PrinterColor = Field(
        default="bold_yellow",
        description="Default color for log messages",
    )
    _printer: Printer = PrivateAttr(default_factory=Printer)

    def log(self, level: str, message: str, color: PrinterColor | None = None) -> None:
        """Log a message with timestamp if verbose mode is enabled.

        Args:
            level: The log level (e.g., 'info', 'warning', 'error').
            message: The message to log.
            color: Optional color for the message. Defaults to default_color.
        """
        if self.verbose:
            timestamp: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._printer.print(
                [
                    ColoredText(f"\n[{timestamp}]", "cyan"),
                    ColoredText(f"[{level.upper()}]: ", "yellow"),
                    ColoredText(message, color or self.default_color),
                ]
            )
