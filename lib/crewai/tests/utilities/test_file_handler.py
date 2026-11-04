import json
import os
from pathlib import Path
from typing import Any
import unittest
import uuid

import pytest
from crewai.utilities.file_handler import FileHandler, PickleHandler


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


@pytest.mark.parametrize("existing_content", ['{"already": "an object"}', '"a bare string"'])
def test_json_log_recovers_when_existing_file_is_not_a_record_list(
    tmp_path: Path, existing_content: str
) -> None:
    """A valid-JSON file that is not a record list must be replaced, not crash every later log call.

    Before this fix, json.load succeeded and .append raised AttributeError, which the
    outer handler converted into ValueError for every subsequent log() call.
    """
    path = tmp_path / "logs.json"
    path.write_text(existing_content, encoding="utf-8")

    handler = FileHandler(str(path))
    handler.log(message="first record")
    handler.log(message="second record")

    records = json.loads(path.read_text(encoding="utf-8"))
    assert [record["message"] for record in records] == ["first record", "second record"]
