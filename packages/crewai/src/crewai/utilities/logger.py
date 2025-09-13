from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from crewai.utilities.printer import Printer


class Logger(BaseModel):
    verbose: bool = Field(default=False)
    _printer: Printer = PrivateAttr(default_factory=Printer)
    default_color: str = Field(default="bold_yellow")

    def log(self, level, message, color=None):
        if color is None:
            color = self.default_color
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._printer.print(
                f"\n[{timestamp}][{level.upper()}]: {message}", color=color
            )
