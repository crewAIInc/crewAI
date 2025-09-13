import os

from crewai.utilities.file_handler import PickleHandler


class CrewTrainingHandler(PickleHandler):
    def save_trained_data(self, agent_id: str, trained_data: dict) -> None:
        """
        Save the trained data for a specific agent.

        Parameters:
        - agent_id (str): The ID of the agent.
        - trained_data (dict): The trained data to be saved.
        """
        data = self.load()
        data[agent_id] = trained_data
        self.save(data)

    def append(self, train_iteration: int, agent_id: str, new_data) -> None:
        """
        Append new data to the existing pickle file.

        Parameters:
        - new_data (object): The new data to be appended.
        """
        data = self.load()

        if agent_id in data:
            data[agent_id][train_iteration] = new_data
        else:
            data[agent_id] = {train_iteration: new_data}

        self.save(data)

    def clear(self) -> None:
        """Clear the training data by removing the file or resetting its contents."""
        if os.path.exists(self.file_path):
            self.save({})
