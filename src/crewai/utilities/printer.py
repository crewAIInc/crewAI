class Printer:
    def print(self, content: str, color: str):
        if color == "yellow":
            self._print_yellow(content)
        elif color == "red":
            self._print_red(content)
        elif color == "bold_green":
            self._print_bold_green(content)
        elif color == "bold_yellow":
            self._print_bold_yellow(content)
        else:
            print(content)

    def _print_bold_yellow(self, content):
        print("\033[1m\033[93m {}\033[00m".format(content))

    def _print_bold_green(self, content):
        print("\033[1m\033[92m {}\033[00m".format(content))

    def _print_yellow(self, content):
        print("\033[93m {}\033[00m".format(content))

    def _print_red(self, content):
        print("\033[91m {}\033[00m".format(content))
