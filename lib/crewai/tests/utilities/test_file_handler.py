import os
import tempfile
import unittest

import pytest
from crewai.utilities.file_handler import PickleHandler


class TestPickleHandler(unittest.TestCase):
    def setUp(self):
        # Use temporary file instead of UUID workaround - thread-safe PickleHandler can handle this
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        self.temp_file.close()
        self.file_path = self.temp_file.name
        # Extract just the filename for PickleHandler
        self.file_name = os.path.basename(self.file_path)
        self.handler = PickleHandler(self.file_name)

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

    def test_load_empty_file(self):
        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_corrupted_file(self):
        with open(self.file_path, "wb") as file:
            file.write(b"corrupted data")
            file.flush()
            os.fsync(file.fileno())  # Ensure data is written to disk

        with pytest.raises(Exception) as exc:
            self.handler.load()

        assert str(exc.value) == "pickle data was truncated"
        assert "<class '_pickle.UnpicklingError'>" == str(exc.type)

    def test_lock_timeout_parameter(self):
        """Test that lock timeout can be configured during initialization."""
        handler_with_timeout = PickleHandler(self.file_name, lock_timeout=2.0)
        assert handler_with_timeout.lock_timeout == 2.0
        
        # Test that it still works with custom timeout
        data = {"timeout_test": "value"}
        handler_with_timeout.save(data)
        loaded_data = handler_with_timeout.load()
        assert loaded_data == data

    def test_thread_safe_operations(self):
        """Test basic thread-safety by ensuring operations complete without errors."""
        # This is a basic test - more comprehensive concurrency tests are in test_pickle_handler_concurrency.py
        data1 = {"test": "data1"}
        data2 = {"test": "data2"}
        
        # Multiple save/load operations should work without issues
        self.handler.save(data1)
        loaded1 = self.handler.load()
        assert loaded1 == data1
        
        self.handler.save(data2)
        loaded2 = self.handler.load()
        assert loaded2 == data2
