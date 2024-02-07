import json
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator


class I18N(BaseModel):
    _translations: Dict[str, Dict[str, str]] = PrivateAttr()
    language: Optional[str] = Field(
        default="en",
        description="Language used to load translations",
    )

    @model_validator(mode="after")
    def load_translation(self) -> "I18N":
        """        Load translations from a JSON file based on the specified language.

        Returns:
            I18N: The instance of the class with loaded translations

        Raises:
            ValidationError: If the translation file for the specified language is not found
            ValidationError: If there is an error decoding JSON from the prompts file
        """
        try:
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
        """        Retrieve the specified slice from the 'slices' collection.

        Args:
            slice (str): The name of the slice to retrieve.

        Returns:
            str: The retrieved slice.
        """

        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        """        Return the error message associated with the given error code.

        Args:
            error (str): The error code for which the error message needs to be retrieved.

        Returns:
            str: The error message associated with the given error code.
        """

        return self.retrieve("errors", error)

    def tools(self, error: str) -> str:
        """        Retrieve tools information.

        Args:
            error (str): The error message.

        Returns:
            str: The retrieved tools information.
        """

        return self.retrieve("tools", error)

    def retrieve(self, kind, key) -> str:
        """        Retrieve the translation for the given kind and key.

        Args:
            kind (str): The kind of translation to retrieve.
            key (str): The key of the translation to retrieve.

        Returns:
            str: The translation corresponding to the given kind and key.

        Raises:
            ValidationError: If the translation for the given kind and key is not found.
        """

        try:
            return self._translations[kind][key]
        except:
            raise ValidationError(f"Translation for '{kind}':'{key}'  not found.")
