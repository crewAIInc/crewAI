import os
import unittest
import uuid

from crewai.utilities.file_handler import JSONHandler


class TestJSONHandler(unittest.TestCase):
    def setUp(self):
        # Use a unique file name for each test to avoid race conditions in parallel test execution
        unique_id = str(uuid.uuid4())
        self.file_name = f"test_data_{unique_id}.json"
        self.file_path = os.path.join(os.getcwd(), self.file_name)
        self.handler = JSONHandler(self.file_name)

    def tearDown(self):
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_initialize_file(self):
        assert os.path.exists(self.file_path) is False

        self.handler.initialize_file()

        assert os.path.exists(self.file_path) is True
        assert os.path.getsize(self.file_path) >= 0

    def test_save_and_load(self):
        data = {"key": "value"}
        self.handler.save(data)
        loaded_data = self.handler.load()
        assert loaded_data == data

    def test_rejects_non_json_filename(self):
        with self.assertRaisesRegex(
            ValueError, "Training data filename must end with .json"
        ):
            JSONHandler("training-data")

    def test_load_empty_file(self):
        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_corrupted_file(self):
        with open(self.file_path, "w", encoding="utf-8") as file:
            file.write("corrupted data")
            file.flush()
            os.fsync(file.fileno())  # Ensure data is written to disk

        assert self.handler.load() == {}
