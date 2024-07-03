import os
import unittest

from crewai.utilities.training_handler import CrewTrainingHandler


class TestCrewTrainingHandler(unittest.TestCase):
    def setUp(self):
        self.handler = CrewTrainingHandler("trained_data.pkl")

    def tearDown(self):
        os.remove("trained_data.pkl")
        del self.handler

    def test_save_trained_data(self):
        agent_id = "agent1"
        trained_data = {"param1": 1, "param2": 2}
        self.handler.save_trained_data(agent_id, trained_data)

        # Assert that the trained data is saved correctly
        data = self.handler.load()
        assert data[agent_id] == trained_data

    def test_append_existing_agent(self):
        train_iteration = 1
        agent_id = "agent1"
        new_data = {"param3": 3, "param4": 4}
        self.handler.append(train_iteration, agent_id, new_data)

        # Assert that the new data is appended correctly to the existing agent
        data = self.handler.load()
        assert data[agent_id][train_iteration] == new_data

    def test_append_new_agent(self):
        train_iteration = 1
        agent_id = "agent2"
        new_data = {"param5": 5, "param6": 6}
        self.handler.append(train_iteration, agent_id, new_data)

        # Assert that the new agent and data are appended correctly
        data = self.handler.load()
        assert data[agent_id][train_iteration] == new_data
