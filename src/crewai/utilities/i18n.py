import json
import os

from pydantic import BaseModel, Field, PrivateAttr, model_validator

"""Internationalization support for CrewAI prompts and messages."""

class I18N(BaseModel):
    """Handles loading and retrieving internationalized prompts."""

    _prompts: dict[str, dict[str, str]] = PrivateAttr()
    prompt_file: str | None = Field(
        default=None,
        description="Path to the prompt_file file to load",
    )

    @model_validator(mode="after")
    def load_prompts(self) -> "I18N":
        """Load prompts from a JSON file."""
        try:
            if self.prompt_file:
                with open(self.prompt_file, encoding="utf-8") as f:
                    self._prompts = json.load(f)
            else:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                prompts_path = os.path.join(dir_path, "../translations/en.json")

                with open(prompts_path, encoding="utf-8") as f:
                    self._prompts = json.load(f)
        except FileNotFoundError:
            msg = f"Prompt file '{self.prompt_file}' not found."
            raise Exception(msg)
        except json.JSONDecodeError:
            msg = "Error decoding JSON from the prompts file."
            raise Exception(msg)

        if not self._prompts:
            self._prompts = {}

        return self

    def slice(self, slice: str) -> str:
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        return self.retrieve("errors", error)

    def tools(self, tool: str) -> str | dict[str, str]:
        return self.retrieve("tools", tool)

    def retrieve(self, kind, key) -> str:
        try:
            return self._prompts[kind][key]
        except Exception as _:
            msg = f"Prompt for '{kind}':'{key}'  not found."
            raise Exception(msg)
