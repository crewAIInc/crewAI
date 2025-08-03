import json
import os
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr, model_validator

"""Internationalization support for CrewAI prompts and messages."""

class I18N(BaseModel):
    """Handles loading and retrieving internationalized prompts."""
    _prompts: Dict[str, Dict[str, str]] = PrivateAttr()
    prompt_file: Optional[str] = Field(
        default=None,
        description="Path to the prompt_file file to load",
    )

    @model_validator(mode="after")
    def load_prompts(self) -> "I18N":
        """Load prompts from a JSON file."""
        prompt_file_to_use = None
        
        try:
            if self.prompt_file:
                prompt_file_to_use = self.prompt_file
            else:
                env_i18n_file = os.environ.get("CREWAI_I18N_FILE")
                if env_i18n_file:
                    prompt_file_to_use = env_i18n_file
                else:
                    dir_path = os.path.dirname(os.path.realpath(__file__))
                    prompt_file_to_use = os.path.join(dir_path, "../translations/en.json")

            with open(prompt_file_to_use, "r", encoding="utf-8") as f:
                self._prompts = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Prompt file '{prompt_file_to_use}' not found.")
        except json.JSONDecodeError:
            raise Exception("Error decoding JSON from the prompts file.")

        if not self._prompts:
            self._prompts = {}

        return self

    def slice(self, slice: str) -> str:
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        return self.retrieve("errors", error)

    def tools(self, tool: str) -> Union[str, Dict[str, str]]:
        return self.retrieve("tools", tool)

    def retrieve(self, kind, key) -> str:
        try:
            return self._prompts[kind][key]
        except Exception as _:
            raise Exception(f"Prompt for '{kind}':'{key}'  not found.")
