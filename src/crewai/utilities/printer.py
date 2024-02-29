class Printer:
    """This class is responsible for printing colored text to the console."""

    def print(self, content: str, color: str):
        """Prints the content in the specified color.

        If the color is 'yellow', it calls the method to print in yellow.
        If the color is 'red', it calls the method to print in red.
        If any other color is specified, it prints the content without any color.
        """
        if color == "yellow":
            self._print_yellow(content)
        elif color == "red":
            self._print_red(content)
        else:
            print(content)

    def _print_yellow(self, content):
        """Prints the content in yellow.

        This is done by wrapping the content in the appropriate ANSI escape codes.
        """
        print("\033[93m {}\033[00m".format(content))

    def _print_red(self, content):
        """Prints the content in red.

        This is done by wrapping the content in the appropriate ANSI escape codes.
        """
        print("\033[91m {}\033[00m".format(content))
