from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from crewai.utilities.printer import Printer


class Logger(BaseModel):
    verbose: bool = Field(default=False)
    _printer: Printer = PrivateAttr(default_factory=Printer)

    def log(self, level, message, color="bold_yellow"):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._printer.print(
                f"\n[{timestamp}][{level.upper()}]: {message}", color=color
            )
