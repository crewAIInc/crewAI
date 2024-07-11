import os
import pickle
import json

from datetime import datetime
from typing import Dict, Any, List

from crewai.utilities.crew_json_encoder import CrewJSONEncoder


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
        with open(self._path, "a", encoding="utf-8") as file:
            file.write(message + "\n")


class PickleHandler:
    def __init__(self, file_name: str) -> None:
        """
        Initialize the PickleHandler with the name of the file where data will be stored.
        The file will be saved in the current directory.

        Parameters:
        - file_name (str): The name of the file for saving and loading data.
        """
        self.file_path = os.path.join(os.getcwd(), file_name)

    def initialize_file(self) -> None:
        """
        Initialize the file with an empty dictionary if it does not exist or is empty.
        """
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            self.save({})  # Save an empty dictionary to initialize the file

    def save(self, data) -> None:
        """
        Save the data to the specified file using pickle.

        Parameters:
        - data (object): The data to be saved.
        """
        with open(self.file_path, "wb") as file:
            pickle.dump(data, file)

    def load(self) -> dict:
        """
        Load the data from the specified file using pickle.

        Returns:
        - dict: The data loaded from the file.
        """
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            return {}  # Return an empty dictionary if the file does not exist or is empty

        with open(self.file_path, "rb") as file:
            try:
                return pickle.load(file)
            except EOFError:
                return {}  # Return an empty dictionary if the file is empty or corrupted
            except Exception:
                raise  # Raise any other exceptions that occur during loading


class TaskOutputJsonHandler:
    def __init__(self, file_name: str) -> None:
        self.file_path = os.path.join(os.getcwd(), file_name)

    def initialize_file(self) -> None:
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            with open(self.file_path, "w") as file:
                json.dump([], file)

    def append(self, log) -> None:
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            # Initialize the file with an empty list if it doesn't exist or is empty
            with open(self.file_path, "w") as file:
                json.dump([], file)
        with open(self.file_path, "r+") as file:
            try:
                file_data = json.load(file)
            except json.JSONDecodeError:
                # If the file contains invalid JSON, initialize it with an empty list
                file_data = []

            file_data.append(log)
            file.seek(0)
            json.dump(file_data, file, indent=2, cls=CrewJSONEncoder)
            file.truncate()

    def update(self, task_index: int, log: Dict[str, Any]):
        logs = self.load()
        if task_index < len(logs):
            logs[task_index] = log
        else:
            logs.append(log)
        self.save(logs)

    def save(self, logs: List[Dict[str, Any]]):
        with open(self.file_path, "w") as file:
            json.dump(logs, file, indent=2, cls=CrewJSONEncoder)

    def reset(self):
        """Reset the JSON file by creating an empty file."""
        with open(self.file_path, "w") as f:
            json.dump([], f)

    def load(self) -> list:
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            return []

        with open(self.file_path, "r") as file:
            return json.load(file)
