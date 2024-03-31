import inspect
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


def CrewBase(cls):
    class WrappedClass(cls):
        is_crew_class = True

        base_directory = None
        for frame_info in inspect.stack():
            if "site-packages" not in frame_info.filename:
                base_directory = Path(frame_info.filename).parent.resolve()
                break

        if base_directory is None:
            raise Exception(
                "Unable to dynamically determine the project's base directory, you must run it from the project's root directory."
            )

        original_agents_config_path = getattr(
            cls, "agents_config", "config/agents.yaml"
        )
        original_tasks_config_path = getattr(cls, "tasks_config", "config/tasks.yaml")

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.agents_config = self.load_yaml(
                os.path.join(self.base_directory, self.original_agents_config_path)
            )
            self.tasks_config = self.load_yaml(
                os.path.join(self.base_directory, self.original_tasks_config_path)
            )

        @staticmethod
        def load_yaml(config_path: str):
            with open(config_path, "r") as file:
                return yaml.safe_load(file)

    return WrappedClass
