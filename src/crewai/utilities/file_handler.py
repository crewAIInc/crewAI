import os
import pickle
from datetime import datetime


class FileHandler:
    """take care of file operations, currently it only logs messages to a file"""
    
    def __init__(self, file_path, save_as_json):
        self.save_as_json = save_as_json
        if file_path is True:  # File path is boolean True
            if save_as_json:
                self._path = os.path.join(os.curdir, "logs.json")
            else:
                self._path = os.path.join(os.curdir, "logs.txt")
        elif isinstance(file_path, str):  # File path is a string
            if save_as_json:
                if not file_path.endswith(".json"):
                    file_path += ".json"
            self._path = file_path
        else:
            raise ValueError("file_path must be either a boolean or a string.")

    def log(self, **kwargs):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {"timestamp": now, **kwargs}

        if self._path.endswith(".json"):
            # Append log in JSON format
            with open(self._path, "a", encoding="utf-8") as file:
                # If the file is empty, start with a list; else, append to it
                try:
                    # Try reading existing content to avoid overwriting
                    with open(self._path, "r", encoding="utf-8") as read_file:
                        existing_data = json.load(read_file)
                        existing_data.append(log_entry)
                except (json.JSONDecodeError, FileNotFoundError):
                    # If no valid JSON or file doesn't exist, start with an empty list
                    existing_data = [log_entry]
                
                with open(self._path, "w", encoding="utf-8") as write_file:
                    json.dump(existing_data, write_file, indent=4)
                    write_file.write("\n")
        else:
            # Append log in plain text format
            message = f"{now}: " + ", ".join([f"{key}=\"{value}\"" for key, value in kwargs.items()]) + "\n"
            with open(self._path, "a", encoding="utf-8") as file:
                file.write(message)



class PickleHandler:
    def __init__(self, file_name: str) -> None:
        """
        Initialize the PickleHandler with the name of the file where data will be stored.
        The file will be saved in the current directory.

        Parameters:
        - file_name (str): The name of the file for saving and loading data.
        """
        if not file_name.endswith(".pkl"):
            file_name += ".pkl"

        self.file_path = os.path.join(os.getcwd(), file_name)

    def initialize_file(self) -> None:
        """
        Initialize the file with an empty dictionary and overwrite any existing data.
        """
        self.save({})

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
                return pickle.load(file)  # nosec
            except EOFError:
                return {}  # Return an empty dictionary if the file is empty or corrupted
            except Exception:
                raise  # Raise any other exceptions that occur during loading
