from typing import Optional
import sys
from enum import Enum


class Color(Enum):
    """Enum for text colors in terminal output."""
    PURPLE = "\033[95m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    BOLD = "\033[1m"
    RESET = "\033[00m"


class Printer:
    """
    Utility class for printing formatted text to stdout.
    Uses direct stdout writing for compatibility with asynchronous environments.
    """
    
    def print(self, content: str, color: Optional[str] = None) -> None:
        """
        Print content with optional color formatting.
        
        Args:
            content: The text to print
            color: Optional color name (e.g., "purple", "bold_green")
        """
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
        
        try:
            sys.stdout.write(f"{output}\n")
            sys.stdout.flush()
        except IOError:
            pass

    def _format_text(self, content: str, color: Color, bold: bool = False) -> str:
        """
        Format text with color and optional bold styling.
        
        Args:
            content: The text to format
            color: The color to apply
            bold: Whether to apply bold formatting
            
        Returns:
            Formatted text string
        """
        if bold:
            return f"{Color.BOLD.value}{color.value} {content}{Color.RESET.value}"
        return f"{color.value} {content}{Color.RESET.value}"

    def _format_bold_purple(self, content: str) -> str:
        """Format text as bold purple."""
        return self._format_text(content, Color.PURPLE, bold=True)

    def _format_bold_green(self, content: str) -> str:
        """Format text as bold green."""
        return self._format_text(content, Color.GREEN, bold=True)

    def _format_purple(self, content: str) -> str:
        """Format text as purple."""
        return self._format_text(content, Color.PURPLE)

    def _format_red(self, content: str) -> str:
        """Format text as red."""
        return self._format_text(content, Color.RED)

    def _format_bold_blue(self, content: str) -> str:
        """Format text as bold blue."""
        return self._format_text(content, Color.BLUE, bold=True)

    def _format_yellow(self, content: str) -> str:
        """Format text as yellow."""
        return self._format_text(content, Color.YELLOW)

    def _format_bold_yellow(self, content: str) -> str:
        """Format text as bold yellow."""
        return self._format_text(content, Color.YELLOW, bold=True)
