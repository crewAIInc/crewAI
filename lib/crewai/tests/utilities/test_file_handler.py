import os
import pickle
import unittest
import uuid

import pytest
from crewai.utilities.file_handler import PickleHandler


class TestPickleHandler(unittest.TestCase):
    def setUp(self):
        # Use a unique file name for each test to avoid race conditions in parallel test execution
        unique_id = str(uuid.uuid4())
        self.file_name = f"test_data_{unique_id}.pkl"
        self.file_path = os.path.join(os.getcwd(), self.file_name)
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

    def test_load_rejects_unsafe_class(self):
        """Verify that RestrictedUnpickler blocks arbitrary code execution.

        A tampered .pkl file could contain a payload that executes arbitrary
        code (e.g., os.system, eval, subprocess.Popen) when loaded with
        unrestricted pickle.load(). The SafeUnpickler should reject any
        class not in its allowlist.
        """
        class _Exploit:
            def __reduce__(self):
                return (eval, ("1+1",))

        payload = pickle.dumps(_Exploit())

        with open(self.file_path, "wb") as f:
            f.write(payload)

        with pytest.raises(pickle.UnpicklingError, match="not in allowlist"):
            self.handler.load()

    def test_load_allows_safe_types(self):
        """Verify that common safe data types can still be loaded."""
        from collections import OrderedDict
        from datetime import datetime

        data = {
            "strings": ["a", "b", "c"],
            "numbers": [1, 2.5, True, None],
            "nested": {"key": (1, 2, 3)},
            "ordered": OrderedDict([("x", 1), ("y", 2)]),
            "timestamp": datetime(2026, 1, 1, 12, 0, 0),
        }
        self.handler.save(data)
        loaded = self.handler.load()
        assert loaded == data
