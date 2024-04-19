import json
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator


class I18N(BaseModel):
    _translations: Dict[str, Dict[str, str]] = PrivateAttr()
    language_file: Optional[str] = Field(
        default=None,
        description="Path to the translation file to load",
    )
    language: Optional[str] = Field(
        default="en",
        description="Language used to load translations",
    )

    @model_validator(mode="after")
    def load_translation(self) -> "I18N":
        """Load translations from a JSON file based on the specified language."""
        try:
            if self.language_file:
                with open(self.language_file, "r") as f:
                    self._translations = json.load(f)
            else:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                prompts_path = os.path.join(
                    dir_path, f"../translations/{self.language}.json"
                )

                with open(prompts_path, "r") as f:
                    self._translations = json.load(f)
        except FileNotFoundError:
            raise ValidationError(
                f"Translation file for language '{self.language}' not found."
            )
        except json.JSONDecodeError:
            raise ValidationError(f"Error decoding JSON from the prompts file.")

        if not self._translations:
            self._translations = {}

        return self

    def slice(self, slice: str) -> str:
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        return self.retrieve("errors", error)

    def tools(self, error: str) -> str:
        return self.retrieve("tools", error)

    def retrieve(self, kind, key) -> str:
        try:
            return self._translations[kind][key]
        except:
            raise ValidationError(f"Translation for '{kind}':'{key}'  not found.")
