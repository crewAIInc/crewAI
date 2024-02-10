class Printer:
    def print(self, content: str, color: str):
        if color == "yellow":
            self._print_yellow(content)
        else:
            print(content)

    def _print_yellow(self, content):
        print("\033[93m {}\033[00m".format(content))
