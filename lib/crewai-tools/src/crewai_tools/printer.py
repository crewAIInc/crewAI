"""Utility for colored console output."""


class Printer:
    """Handles colored console output formatting."""

    @staticmethod
    def print(content: str, color: str | None = None) -> None:
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
            print(content)  # noqa: T201

    @staticmethod
    def _print_bold_purple(content: str) -> None:
        """Prints content in bold purple color.

        Args:
            content: The string to be printed in bold purple.
        """
        print(f"\033[1m\033[95m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_bold_green(content: str) -> None:
        """Prints content in bold green color.

        Args:
            content: The string to be printed in bold green.
        """
        print(f"\033[1m\033[92m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_purple(content: str) -> None:
        """Prints content in purple color.

        Args:
            content: The string to be printed in purple.
        """
        print(f"\033[95m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_red(content: str) -> None:
        """Prints content in red color.

        Args:
            content: The string to be printed in red.
        """
        print(f"\033[91m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_bold_blue(content: str) -> None:
        """Prints content in bold blue color.

        Args:
            content: The string to be printed in bold blue.
        """
        print(f"\033[1m\033[94m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_yellow(content: str) -> None:
        """Prints content in yellow color.

        Args:
            content: The string to be printed in yellow.
        """
        print(f"\033[93m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_bold_yellow(content: str) -> None:
        """Prints content in bold yellow color.

        Args:
            content: The string to be printed in bold yellow.
        """
        print(f"\033[1m\033[93m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_cyan(content: str) -> None:
        """Prints content in cyan color.

        Args:
            content: The string to be printed in cyan.
        """
        print(f"\033[96m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_bold_cyan(content: str) -> None:
        """Prints content in bold cyan color.

        Args:
            content: The string to be printed in bold cyan.
        """
        print(f"\033[1m\033[96m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_magenta(content: str) -> None:
        """Prints content in magenta color.

        Args:
            content: The string to be printed in magenta.
        """
        print(f"\033[35m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_bold_magenta(content: str) -> None:
        """Prints content in bold magenta color.

        Args:
            content: The string to be printed in bold magenta.
        """
        print(f"\033[1m\033[35m {content}\033[00m")  # noqa: T201

    @staticmethod
    def _print_green(content: str) -> None:
        """Prints content in green color.

        Args:
            content: The string to be printed in green.
        """
        print(f"\033[32m {content}\033[00m")  # noqa: T201
