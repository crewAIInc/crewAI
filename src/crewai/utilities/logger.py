from datetime import datetime

from crewai.utilities.printer import Printer


class Logger:
    _printer = Printer()

    def log(self, level, message, color="bold_green"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._printer.print(f"[{timestamp}][{level.upper()}]: {message}", color=color)
