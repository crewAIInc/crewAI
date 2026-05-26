import os
import tempfile
import unittest
from unittest.mock import patch

from crewai.utilities.training_handler import CrewTrainingHandler


class InternalCrewTrainingHandler(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        self.temp_file.close()
        self.handler = CrewTrainingHandler(self.temp_file.name)

    def tearDown(self):
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
        del self.handler

    def test_save_trained_data(self):
        agent_id = "agent1"
        trained_data = {"param1": 1, "param2": 2}
        self.handler.save_trained_data(agent_id, trained_data)

        # Assert that the trained data is saved correctly
        data = self.handler.load()
        assert data[agent_id] == trained_data

    def test_append_existing_agent(self):
        agent_id = "agent1"
        initial_iteration = 0
        initial_data = {"param1": 1, "param2": 2}

        self.handler.append(initial_iteration, agent_id, initial_data)

        train_iteration = 1
        new_data = {"param3": 3, "param4": 4}
        self.handler.append(train_iteration, agent_id, new_data)

        # Assert that the new data is appended correctly to the existing agent
        data = self.handler.load()
        assert agent_id in data
        assert initial_iteration in data[agent_id]
        assert train_iteration in data[agent_id]
        assert data[agent_id][initial_iteration] == initial_data
        assert data[agent_id][train_iteration] == new_data

    def test_append_new_agent(self):
        train_iteration = 1
        agent_id = "agent2"
        new_data = {"param5": 5, "param6": 6}
        self.handler.append(train_iteration, agent_id, new_data)

        # Assert that the new agent and data are appended correctly
        data = self.handler.load()
        assert data[agent_id][train_iteration] == new_data

    def test_load_missing_file_does_not_acquire_lock(self):
        handler = CrewTrainingHandler(self.temp_file.name + ".missing")

        with patch(
            "crewai.utilities.file_handler.store_lock",
            side_effect=AssertionError("load() acquired lock for missing file"),
        ):
            assert handler.load() == {}

    def test_load_acquires_lock_for_zero_size_file(self):
        # Empty file mimics a concurrent save() mid-truncation (open "wb").
        assert os.path.getsize(self.temp_file.name) == 0

        with patch(
            "crewai.utilities.file_handler.store_lock",
            side_effect=AssertionError("load() short-circuited on size 0"),
        ):
            with self.assertRaises(AssertionError):
                self.handler.load()
