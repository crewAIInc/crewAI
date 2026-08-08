import io
import os
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
        sig_path = self.file_path + ".sig"
        if os.path.exists(sig_path):
            os.remove(sig_path)

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

    def test_save_creates_signature_file(self):
        data = {"key": "value"}
        self.handler.save(data)
        sig_path = self.file_path + ".sig"
        assert os.path.exists(sig_path)
        assert os.path.getsize(sig_path) == 32  # SHA-256 digest size

    def test_load_empty_file(self):
        loaded_data = self.handler.load()
        assert loaded_data == {}

    def test_load_corrupted_file(self):
        with open(self.file_path, "wb") as file:
            file.write(b"corrupted data")
            file.flush()
            os.fsync(file.fileno())

        with pytest.raises(Exception) as exc:
            self.handler.load()

        assert str(exc.value) == "pickle data was truncated"
        assert "<class '_pickle.UnpicklingError'>" == str(exc.type)

    def test_load_tampered_file_raises_error(self):
        data = {"key": "value"}
        self.handler.save(data)

        # Tamper with the pickle file
        with open(self.file_path, "wb") as f:
            f.write(b"tampered pickle data")

        with pytest.raises(ValueError, match="Integrity check failed"):
            self.handler.load()

    def test_load_legacy_file_without_signature(self):
        # Write a pickle file without a signature (simulates pre-fix files)
        import pickle

        with open(self.file_path, "wb") as f:
            pickle.dump({"legacy": True}, f)

        # Should load with a warning, not raise
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded_data = self.handler.load()
            assert loaded_data == {"legacy": True}
            assert len(w) == 1
            assert "signature file" in str(w[0].message)

    def test_overwrite_preserves_signature(self):
        data1 = {"first": True}
        self.handler.save(data1)
        loaded1 = self.handler.load()
        assert loaded1 == data1

        data2 = {"second": True}
        self.handler.save(data2)
        loaded2 = self.handler.load()
        assert loaded2 == data2

    def test_initialize_file_creates_valid_signature(self):
        self.handler.initialize_file()
        sig_path = self.file_path + ".sig"
        assert os.path.exists(sig_path)

        # Should load without warning since signature exists
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded_data = self.handler.load()
            assert loaded_data == {}
            assert len(w) == 0
