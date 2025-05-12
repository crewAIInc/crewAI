import os
import unittest

import pytest

from crewai.utilities.file_handler import PickleHandler


class TestPickleHandler(unittest.TestCase):
    def setUp(self) -> None:
        self.file_name = "test_data.pkl"
        self.file_path = os.path.join(os.getcwd(), self.file_name)
        self.handler = PickleHandler(self.file_name)

    def tearDown(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_initialize_file(self) -> None:
        assert os.path.exists(self.file_path) is False

        self.handler.initialize_file()

        assert os.path.exists(self.file_path) is True
        assert os.path.getsize(self.file_path) >= 0

    def test_save_and_load(self) -> None:
        data = {"key": "value"}
        self.handler.save(data)
        loaded_data = self.handler.load()
        assert loaded_data == data

    def test_load_empty_file(self) -> None:
        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_corrupted_file(self) -> None:
        with open(self.file_path, "wb") as file:
            file.write(b"corrupted data")

        with pytest.raises(Exception) as exc:
            self.handler.load()

        assert str(exc.value) == "pickle data was truncated"
        assert str(exc.type) == "<class '_pickle.UnpicklingError'>"
