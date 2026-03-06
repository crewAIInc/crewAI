import os
import tempfile
import unittest

from crewai.utilities.training_handler import CrewTrainingHandler


class InternalCrewTrainingHandler(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.temp_file.close()
        self.handler = CrewTrainingHandler(self.temp_file.name)

    def tearDown(self):
        # Clean up both potential file paths (.json used by handler)
        handler_path = self.handler.file_path
        for path in (self.temp_file.name, handler_path):
            if os.path.exists(path):
                os.remove(path)
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
        # Note: JSON serializes integer keys as strings
        data = self.handler.load()
        assert agent_id in data
        assert str(initial_iteration) in data[agent_id]
        assert str(train_iteration) in data[agent_id]
        assert data[agent_id][str(initial_iteration)] == initial_data
        assert data[agent_id][str(train_iteration)] == new_data

    def test_append_new_agent(self):
        train_iteration = 1
        agent_id = "agent2"
        new_data = {"param5": 5, "param6": 6}
        self.handler.append(train_iteration, agent_id, new_data)

        # Assert that the new agent and data are appended correctly
        # Note: JSON serializes integer keys as strings
        data = self.handler.load()
        assert data[agent_id][str(train_iteration)] == new_data
