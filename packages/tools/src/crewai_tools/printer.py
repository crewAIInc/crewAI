"""Utility for colored console output."""

from typing import Optional


class Printer:
    """Handles colored console output formatting."""

    @staticmethod
    def print(content: str, color: Optional[str] = None) -> None:
        """Prints content with optional color formatting.

        Args:
            content: The string to be printed.
            color: Optional color name to format the output. If provided,
                must match one of the _print_* methods available in this class.
                If not provided or if the color is not supported, prints without
                formatting.
        """
        if hasattr(Printer, f"_print_{color}"):
            getattr(Printer, f"_print_{color}")(content)
        else:
            print(content)

    @staticmethod
    def _print_bold_purple(content: str) -> None:
        """Prints content in bold purple color.

        Args:
            content: The string to be printed in bold purple.
        """
        print("\033[1m\033[95m {}\033[00m".format(content))

    @staticmethod
    def _print_bold_green(content: str) -> None:
        """Prints content in bold green color.

        Args:
            content: The string to be printed in bold green.
        """
        print("\033[1m\033[92m {}\033[00m".format(content))

    @staticmethod
    def _print_purple(content: str) -> None:
        """Prints content in purple color.

        Args:
            content: The string to be printed in purple.
        """
        print("\033[95m {}\033[00m".format(content))

    @staticmethod
    def _print_red(content: str) -> None:
        """Prints content in red color.

        Args:
            content: The string to be printed in red.
        """
        print("\033[91m {}\033[00m".format(content))

    @staticmethod
    def _print_bold_blue(content: str) -> None:
        """Prints content in bold blue color.

        Args:
            content: The string to be printed in bold blue.
        """
        print("\033[1m\033[94m {}\033[00m".format(content))

    @staticmethod
    def _print_yellow(content: str) -> None:
        """Prints content in yellow color.

        Args:
            content: The string to be printed in yellow.
        """
        print("\033[93m {}\033[00m".format(content))

    @staticmethod
    def _print_bold_yellow(content: str) -> None:
        """Prints content in bold yellow color.

        Args:
            content: The string to be printed in bold yellow.
        """
        print("\033[1m\033[93m {}\033[00m".format(content))

    @staticmethod
    def _print_cyan(content: str) -> None:
        """Prints content in cyan color.

        Args:
            content: The string to be printed in cyan.
        """
        print("\033[96m {}\033[00m".format(content))

    @staticmethod
    def _print_bold_cyan(content: str) -> None:
        """Prints content in bold cyan color.

        Args:
            content: The string to be printed in bold cyan.
        """
        print("\033[1m\033[96m {}\033[00m".format(content))

    @staticmethod
    def _print_magenta(content: str) -> None:
        """Prints content in magenta color.

        Args:
            content: The string to be printed in magenta.
        """
        print("\033[35m {}\033[00m".format(content))

    @staticmethod
    def _print_bold_magenta(content: str) -> None:
        """Prints content in bold magenta color.

        Args:
            content: The string to be printed in bold magenta.
        """
        print("\033[1m\033[35m {}\033[00m".format(content))

    @staticmethod
    def _print_green(content: str) -> None:
        """Prints content in green color.

        Args:
            content: The string to be printed in green.
        """
        print("\033[32m {}\033[00m".format(content))
