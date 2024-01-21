import json
import os
from typing import Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator


class I18N(BaseModel):
    _translations: Optional[Dict[str, str]] = PrivateAttr()
    language: Optional[str] = Field(
        default="en",
        description="Language used to load translations",
    )

    @model_validator(mode="after")
    def load_translation(self) -> "I18N":
        """Load translations from a JSON file based on the specified language."""
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            prompts_path = os.path.join(
                dir_path, f"../translations/i18n.json"
            )

            with open(prompts_path, "r") as f:
                all_translations = json.load(f)
                self._translations = all_translations.get(self.language)
        except FileNotFoundError:
            raise ValidationError(
                f"Trasnlation file for language i18n.json not found."
            )
        except json.JSONDecodeError:
            raise ValidationError(f"Error decoding JSON from the prompts file.")
        except TypeError:
            raise ValidationError(f"No translations found for language '{self.language}'.")        
        return self

    def slice(self, slice: str) -> str:
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        return self.retrieve("errors", error)

    def tools(self, tool: str) -> str:
        return self.retrieve("tools", tool)

    def retrieve(self, kind, key):
        try:
            return self._translations[kind].get(key)
        except:
            raise ValidationError(f"Translation for '{kind}':'{key}'  not found.")
