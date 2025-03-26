from typing import Optional
import sys


class Printer:
    def print(self, content: str, color: Optional[str] = None):
        output = content
        if color == "purple":
            output = self._format_purple(content)
        elif color == "red":
            output = self._format_red(content)
        elif color == "bold_green":
            output = self._format_bold_green(content)
        elif color == "bold_purple":
            output = self._format_bold_purple(content)
        elif color == "bold_blue":
            output = self._format_bold_blue(content)
        elif color == "yellow":
            output = self._format_yellow(content)
        elif color == "bold_yellow":
            output = self._format_bold_yellow(content)
        sys.stdout.write(f"{output}\n")
        sys.stdout.flush()

    def _format_bold_purple(self, content):
        return "\033[1m\033[95m {}\033[00m".format(content)

    def _format_bold_green(self, content):
        return "\033[1m\033[92m {}\033[00m".format(content)

    def _format_purple(self, content):
        return "\033[95m {}\033[00m".format(content)

    def _format_red(self, content):
        return "\033[91m {}\033[00m".format(content)

    def _format_bold_blue(self, content):
        return "\033[1m\033[94m {}\033[00m".format(content)

    def _format_yellow(self, content):
        return "\033[93m {}\033[00m".format(content)

    def _format_bold_yellow(self, content):
        return "\033[1m\033[93m {}\033[00m".format(content)
