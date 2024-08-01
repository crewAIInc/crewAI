class ContextLengthExceeded(Exception):
    def __init__(self, exceptions):
        self.exceptions = exceptions
        super().__init__(self.__str__())

    def __str__(self):
        error_messages = [str(e) for e in self.exceptions]
        return f"Multiple BadRequestExceptions occurred: {', '.join(error_messages)}"
