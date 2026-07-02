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

    def test_load_round_trips_training_data_artifact_shape(self):
        data = {
            "agent_id": {
                "0": {
                    "initial_output": "Initial output",
                    "human_feedback": "Human feedback",
                    "improved_output": "Improved output",
                }
            }
        }

        self.handler.save(data)

        assert self.handler.load() == data

    def test_load_round_trips_trained_agents_artifact_shape(self):
        data = {
            "researcher": {
                "suggestions": [
                    "Use precise terminology.",
                    "Explain assumptions before giving the answer.",
                ],
                "quality": 8.0,
                "final_summary": "The agent improved after applying feedback.",
            }
        }

        self.handler.save(data)

        assert self.handler.load() == data

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

    def test_load_rejects_unsafe_pickle_globals(self):
        marker = f"CREWAI_PICKLE_HANDLER_EXPLOITED_{uuid.uuid4().hex}"
        previous_value = os.environ.get(marker)

        class _Exploit:
            def __reduce__(self):
                return (exec, (f"import os; os.environ[{marker!r}] = '1'",))

        with open(self.file_path, "wb") as file:
            pickle.dump(_Exploit(), file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())

        try:
            with pytest.raises(pickle.UnpicklingError, match="Refusing to unpickle"):
                self.handler.load()

            assert marker not in os.environ
        finally:
            if previous_value is None:
                os.environ.pop(marker, None)
            else:
                os.environ[marker] = previous_value
