import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "crewai" / "settings.json"


class Settings(BaseModel):
    tool_repository_username: Optional[str] = Field(
        None, description="Username for interacting with the Tool Repository"
    )
    tool_repository_password: Optional[str] = Field(
        None, description="Password for interacting with the Tool Repository"
    )
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH, exclude=True)

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH, **data):
        """Load Settings from config path"""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        file_data = {}
        if config_path.is_file():
            try:
                with config_path.open("r") as f:
                    file_data = json.load(f)
            except json.JSONDecodeError:
                file_data = {}

        merged_data = {**file_data, **data}
        super().__init__(config_path=config_path, **merged_data)

    def dump(self) -> None:
        """Save current settings to settings.json"""
        if self.config_path.is_file():
            with self.config_path.open("r") as f:
                existing_data = json.load(f)
        else:
            existing_data = {}

        updated_data = {**existing_data, **self.model_dump(exclude_unset=True)}
        with self.config_path.open("w") as f:
            json.dump(updated_data, f, indent=4)
