class Printer:
    def print(self, content: str, color: str):
        if color == "purple":
            self._print_purple(content)
        elif color == "red":
            self._print_red(content)
        elif color == "bold_green":
            self._print_bold_green(content)
        elif color == "bold_purple":
            self._print_bold_purple(content)
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
