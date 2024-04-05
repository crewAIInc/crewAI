import os
from datetime import datetime


class FileHandler:
    """take care of file operations, currently it only logs messages to a file"""

    def __init__(self, file_path):
        if isinstance(file_path, bool):
            self._path = os.path.join(os.curdir, "logs.txt")
        elif isinstance(file_path, str):
            self._path = file_path
        else:
            raise ValueError("file_path must be either a boolean or a string.")

    def log(self, **kwargs):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"{now}: ".join([f"{key}={value}" for key, value in kwargs.items()])
        with open(self._path, "a") as file:
            file.write(message + "\n")
