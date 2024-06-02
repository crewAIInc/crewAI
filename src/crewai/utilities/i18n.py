import json
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator


class I18N(BaseModel):
    _prompts: Dict[str, Dict[str, str]] = PrivateAttr()
    prompt_file: Optional[str] = Field(
        default=None,
        description="Path to the prompt_file file to load",
    )
    prompt_overrides: Optional[dict] = Field(
        default=None,
        description="dict with overrides to the prompts loaded from the prompt_file file",
    )
    
    @classmethod
    def merge_dicts(cls, d1, d2):
        for key, value in d2.items():
            if isinstance(value, dict) and key in d1 and isinstance(d1[key], dict):
                cls.merge_dicts(d1[key], value)
            else:
                d1[key] = value
        return d1

    @model_validator(mode="after")
    def load_prompts(self) -> "I18N":
        """Load prompts from a JSON file."""
        try:
            if self.prompt_file:
                with open(self.prompt_file, "r") as f:
                    self._prompts = json.load(f)
            else:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                prompts_path = os.path.join(dir_path, "../translations/en.json")

                with open(prompts_path, "r") as f:
                    self._prompts = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Prompt file '{self.prompt_file}' not found.")
        except json.JSONDecodeError:
            raise Exception("Error decoding JSON from the prompts file.")

        if not self._prompts:
            self._prompts = {}
        
        if self.prompt_overrides:
            self.merge_dicts(self._prompts, self.prompt_overrides)
            
        return self

    def slice(self, slice: str) -> str:
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        return self.retrieve("errors", error)

    def tools(self, error: str) -> str:
        return self.retrieve("tools", error)

    def retrieve(self, kind, key) -> str:
        try:
            return self._prompts[kind][key]
        except Exception as _:
            raise Exception(f"Prompt for '{kind}':'{key}'  not found.")
