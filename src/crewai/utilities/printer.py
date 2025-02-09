"""Utility for colored console output."""

from typing import Optional


class Printer:
    """Handles colored console output formatting."""

    def print(self, content: str, color: Optional[str] = None):
        if color == "purple":
            self._print_purple(content)
        elif color == "red":
            self._print_red(content)
        elif color == "bold_green":
            self._print_bold_green(content)
        elif color == "bold_purple":
            self._print_bold_purple(content)
        elif color == "bold_blue":
            self._print_bold_blue(content)
        elif color == "yellow":
            self._print_yellow(content)
        elif color == "bold_yellow":
            self._print_bold_yellow(content)
        elif color == "cyan":
            self._print_cyan(content)
        elif color == "bold_cyan":
            self._print_bold_cyan(content)
        elif color == "magenta":
            self._print_magenta(content)
        elif color == "bold_magenta":
            self._print_bold_magenta(content)
        elif color == "green":
            self._print_green(content)
        else:
            print(content)

    def _print_bold_purple(self, content):
        print("\033[1m\033[95m {}\033[00m".format(content))

    def _print_bold_green(self, content):
        print("\033[1m\033[92m {}\033[00m".format(content))

    def _print_purple(self, content):
        print("\033[95m {}\033[00m".format(content))

    def _print_red(self, content):
        print("\033[91m {}\033[00m".format(content))

    def _print_bold_blue(self, content):
        print("\033[1m\033[94m {}\033[00m".format(content))

    def _print_yellow(self, content):
        print("\033[93m {}\033[00m".format(content))

    def _print_bold_yellow(self, content):
        print("\033[1m\033[93m {}\033[00m".format(content))

    def _print_cyan(self, content):
        print("\033[96m {}\033[00m".format(content))

    def _print_bold_cyan(self, content):
        print("\033[1m\033[96m {}\033[00m".format(content))

    def _print_magenta(self, content):
        print("\033[35m {}\033[00m".format(content))

    def _print_bold_magenta(self, content):
        print("\033[1m\033[35m {}\033[00m".format(content))

    def _print_green(self, content):
        print("\033[32m {}\033[00m".format(content))
