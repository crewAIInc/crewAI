import json
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator


class I18N(BaseModel):
    """This class is responsible for handling internationalization (i18n) in the application."""

    _translations: Dict[str, Dict[str, str]] = PrivateAttr()  # Private attribute to store loaded translations
    language: Optional[str] = Field(
        default="en",
        description="The language code used to load the appropriate translations file",
    )

    @model_validator(mode="after")
    def load_translation(self) -> "I18N":
        """Load translations from a JSON file based on the specified language.

        This method is called after the model validation. It tries to open a JSON file that matches the specified
        language code. If the file is found, it loads the translations into the _translations attribute. If the file
        is not found or the JSON is not valid, it raises a ValidationError.
        """
        try:
            # Get the directory path of the current file
            dir_path = os.path.dirname(os.path.realpath(__file__))
            # Construct the path to the translations file
            prompts_path = os.path.join(
                dir_path, f"../translations/{self.language}.json"
            )

            # Open the translations file and load the JSON content
            with open(prompts_path, "r") as f:
                self._translations = json.load(f)
        except FileNotFoundError:
            # If the file is not found, raise a ValidationError
            raise ValidationError(
                f"Translation file for language '{self.language}' not found."
            )
        except json.JSONDecodeError:
            # If the JSON is not valid, raise a ValidationError
            raise ValidationError(f"Error decoding JSON from the prompts file.")

        # If no translations were loaded, initialize _translations as an empty dictionary
        if not self._translations:
            self._translations = {}

        return self

    def slice(self, slice: str) -> str:
        """Retrieve a specific slice translation."""
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        """Retrieve a specific error message translation."""
        return self.retrieve("errors", error)

    def tools(self, error: str) -> str:
        """Retrieve a specific tool name translation."""
        return self.retrieve("tools", error)

    def retrieve(self, kind, key) -> str:
        """Retrieve a specific translation based on its kind and key.

        This method tries to retrieve a translation from the _translations attribute using the provided kind and key.
        If the translation is not found, it raises a ValidationError.
        """
        try:
            return self._translations[kind][key]
        except:
            raise ValidationError(f"Translation for '{kind}':'{key}'  not found.")
