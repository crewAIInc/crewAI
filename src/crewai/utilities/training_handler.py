import os
from typing import Any

from crewai.utilities.file_handler import PickleHandler


class CrewTrainingHandler(PickleHandler):
    def save_trained_data(self, agent_id: str, trained_data: dict[int, Any]) -> None:
        """Save the trained data for a specific agent.

        Args:
            agent_id: The ID of the agent.
            trained_data: The trained data to be saved.
        """
        data = self.load()
        data[agent_id] = trained_data
        self.save(data)

    def clear(self) -> None:
        """Clear the training data by removing the file or resetting its contents."""
        if os.path.exists(self.file_path):
            self.save({})
