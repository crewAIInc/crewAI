import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any
import unittest
import uuid
from unittest.mock import patch

import pytest
from crewai.utilities.file_handler import FileHandler, PickleHandler


class TestPickleHandler(unittest.TestCase):
    def setUp(self):
        # Use a unique file name for each test to avoid race conditions in parallel test execution
        # Isolate HOME so HMAC key storage never touches the real home directory
        self._home_patcher = patch.dict(os.environ, {})
        self._home_patcher.start()

        self._tmp_home = tempfile.mkdtemp(prefix="crewai_test_home_")
        self._home_patch = patch("os.path.expanduser", return_value=self._tmp_home)
        self._home_patch.start()

        unique_id = str(uuid.uuid4())
        self.file_name = f"test_data_{unique_id}.pkl"
        self.file_path = os.path.join(os.getcwd(), self.file_name)
        self.handler = PickleHandler(self.file_name)

    def tearDown(self):
        self._home_patch.stop()
        self._home_patcher.stop()

        if os.path.exists(self.file_path):
            os.remove(self.file_path)
        sig_path = self.file_path + ".sig"
        if os.path.exists(sig_path):
            os.remove(sig_path)

        import shutil

        shutil.rmtree(self._tmp_home, ignore_errors=True)

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

    def test_load_rejects_disappearing_signature(self):
        """If the signature file is removed after the existence check, load should raise ValueError."""
        data = {"key": "value"}
        self.handler.save(data)

        original_exists = os.path.exists
        sig_path = self.file_path + ".sig"

        def remove_sig_after_check(path):
            if path == sig_path and original_exists(path):
                os.remove(sig_path)
                return True  # exists() must report True so open() hits FileNotFoundError
            return original_exists(path)

        with patch("os.path.exists", side_effect=remove_sig_after_check):
            with pytest.raises(ValueError, match="signature file"):
                self.handler.load()


@pytest.mark.parametrize("kind", ["unsupported", "circular"])
@pytest.mark.parametrize("existing", [False, True])
def test_json_log_serialization_failure_preserves_file(
    tmp_path: Path, kind: str, existing: bool
) -> None:
    path = tmp_path / "logs.json"
    handler = FileHandler(str(path))
    original = None
    if existing:
        handler.log(message="first record")
        original = path.read_bytes()

    metadata: dict[str, Any] = {}
    if kind == "circular":
        metadata["self"] = metadata
    else:
        metadata["unsupported"] = object()

    with pytest.raises(ValueError, match="Failed to log message"):
        handler.log(message="failed record", metadata=metadata)

    if existing:
        assert path.read_bytes() == original
    else:
        assert not path.exists()

    handler.log(message="next record")
    records = json.loads(path.read_text(encoding="utf-8"))
    expected = ["first record", "next record"] if existing else ["next record"]
    assert [record["message"] for record in records] == expected


def test_json_log_appends_serializable_metadata(tmp_path: Path) -> None:
    path = tmp_path / "logs.json"
    handler = FileHandler(str(path))
    handler.log(message="first", metadata={"count": 1})
    handler.log(message="second", metadata={"labels": ["a", "b"]})

    records = json.loads(path.read_text(encoding="utf-8"))
    assert [record["message"] for record in records] == ["first", "second"]
    assert records[0]["metadata"] == {"count": 1}
    assert records[1]["metadata"] == {"labels": ["a", "b"]}
    assert path.read_text(encoding="utf-8").endswith("\n")
