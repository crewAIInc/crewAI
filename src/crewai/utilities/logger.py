from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from crewai.utilities.printer import Printer


class Logger(BaseModel):
    verbose: bool = Field(default=False)
    _printer: Printer = PrivateAttr(default_factory=Printer)

    def log(self, level, message, color="bold_yellow"):
        if self.verbose or level.upper() in ["WARNING", "ERROR"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._printer.print(
                f"\n[{timestamp}][{level.upper()}]: {message}", color=color
            )

    def debug(self, message: str) -> None:
        """Log a debug message if verbose is enabled."""
        self.log("debug", message, color="bold_blue")

    def info(self, message: str) -> None:
        """Log an info message if verbose is enabled."""
        self.log("info", message, color="bold_green")

    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.log("warning", message, color="bold_yellow")

    def error(self, message: str) -> None:
        """Log an error message."""
        self.log("error", message, color="bold_red")
