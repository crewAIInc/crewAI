from datetime import datetime

from crewai.utilities.printer import Printer


class Logger:
    _printer = Printer()

    def __init__(self, verbose=False):
        self.verbose = verbose

    def log(self, level, message, color="bold_green"):
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._printer.print(
                f"[{timestamp}][{level.upper()}]: {message}", color=color
            )
