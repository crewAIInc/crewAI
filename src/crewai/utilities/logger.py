from crewai.utilities.printer import Printer


class Logger:
    _printer = Printer()

    def __init__(self, verbose_level=0):
        verbose_level = (
            2 if isinstance(verbose_level, bool) and verbose_level else verbose_level
        )
        self.verbose_level = verbose_level

    def log(self, level, message, color="bold_green"):
        level_map = {"debug": 1, "info": 2}
        if self.verbose_level and level_map.get(level, 0) <= self.verbose_level:
            self._printer.print(f"[{level.upper()}]: {message}", color=color)
