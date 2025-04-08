import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr, model_validator

"""Internationalization support for CrewAI prompts and messages."""

SUPPORTED_LANGUAGES = Literal["en", "fr", "es", "pt"]

class I18N(BaseModel):
    """Handles loading and retrieving internationalized prompts."""
    _prompts: Dict[str, Dict[str, str]] = PrivateAttr()
    prompt_file: Optional[str] = Field(
        default=None,
        description="Path to the prompt_file file to load",
    )
    language: Optional[str] = Field(
        default="en",
        description="Language to use for translations. Defaults to English.",
    )
    
    @model_validator(mode="before")
    @classmethod
    def validate_language(cls, data):
        """
        Validate the language parameter.
        
        If the language is not supported, it will fall back to English.
        """
        if isinstance(data, dict) and "language" in data:
            lang = data["language"]
            if lang and lang not in ["en", "fr", "es", "pt"]:
                print(f"Warning: Language '{lang}' not supported. Falling back to English.")
                data["language"] = "en"
        return data

    @model_validator(mode="after")
    def load_prompts(self) -> "I18N":
        """
        Load prompts from a JSON file.
        
        If prompt_file is provided, loads from that file.
        Otherwise, attempts to load from the language-specific translation file.
        Falls back to English if the specified language file doesn't exist.
        
        Raises:
            Exception: If the prompt file is not found or contains invalid JSON.
        
        Returns:
            I18N: The instance with loaded prompts.
        """
        try:
            if self.prompt_file:
                with open(self.prompt_file, "r", encoding="utf-8") as f:
                    self._prompts = json.load(f)
            else:
                base_path = Path(__file__).parent / "../translations"
                lang = self.language or "en"
                lang_file = base_path / f"{lang}.json"
                
                if not lang_file.exists():
                    lang_file = base_path / "en.json"
                
                with open(lang_file.resolve(), "r", encoding="utf-8") as f:
                    self._prompts = json.load(f)
        except FileNotFoundError:
            raise Exception(f"Prompt file '{self.prompt_file or lang_file}' not found.")
        except json.JSONDecodeError:
            raise Exception("Error decoding JSON from the prompts file.")

        if not self._prompts:
            self._prompts = {}

        return self

    def slice(self, slice: str) -> str:
        """Get a slice prompt by key."""
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        """Get an error message by key."""
        return self.retrieve("errors", error)

    def tools(self, tool: str) -> Union[str, Dict[str, str]]:
        """Get a tool prompt by key."""
        return self.retrieve("tools", tool)

    def retrieve(self, kind: str, key: str) -> str:
        """
        Retrieve a prompt by section and key.
        
        Args:
            kind: The section in the prompts file (e.g., "slices", "errors")
            key: The specific key within the section
            
        Returns:
            The prompt text
            
        Raises:
            Exception: If the prompt is not found
        """
        try:
            return self._prompts[kind][key]
        except Exception as _:
            raise Exception(f"Prompt for '{kind}':'{key}'  not found.")
