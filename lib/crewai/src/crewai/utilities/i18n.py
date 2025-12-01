"""Internationalization support for CrewAI prompts and messages."""

from functools import lru_cache
import json
import os
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from typing_extensions import Self


class I18N(BaseModel):
    """Handles loading and retrieving internationalized prompts.

    Attributes:
        _prompts: Internal dictionary storing loaded prompts.
        prompt_file: Optional path to a custom JSON file containing prompts.
    """

    _prompts: dict[str, dict[str, str]] = PrivateAttr()
    prompt_file: str | None = Field(
        default=None,
        description="Path to the prompt_file file to load",
    )

    @model_validator(mode="after")
    def load_prompts(self) -> Self:
        """Load prompts from a JSON file.

        Returns:
            The I18N instance with loaded prompts.

        Raises:
            Exception: If the prompt file is not found or cannot be decoded.
        """
        try:
            if self.prompt_file:
                with open(self.prompt_file, encoding="utf-8") as f:
                    self._prompts = json.load(f)
            else:
                dir_path = os.path.dirname(os.path.realpath(__file__))
                prompts_path = os.path.join(dir_path, "../translations/en.json")

                with open(prompts_path, encoding="utf-8") as f:
                    self._prompts = json.load(f)
        except FileNotFoundError as e:
            raise Exception(f"Prompt file '{self.prompt_file}' not found.") from e
        except json.JSONDecodeError as e:
            raise Exception("Error decoding JSON from the prompts file.") from e

        if not self._prompts:
            self._prompts = {}

        return self

    def slice(self, slice: str) -> str:
        """Retrieve a prompt slice by key.

        Args:
            slice: The key of the prompt slice to retrieve.

        Returns:
            The prompt slice as a string.
        """
        return self.retrieve("slices", slice)

    def errors(self, error: str) -> str:
        """Retrieve an error message by key.

        Args:
            error: The key of the error message to retrieve.

        Returns:
            The error message as a string.
        """
        return self.retrieve("errors", error)

    def tools(self, tool: str) -> str | dict[str, str]:
        """Retrieve a tool prompt by key.

        Args:
            tool: The key of the tool prompt to retrieve.

        Returns:
            The tool prompt as a string or dictionary.
        """
        return self.retrieve("tools", tool)

    def retrieve(
        self,
        kind: Literal[
            "slices", "errors", "tools", "reasoning", "hierarchical_manager_agent"
        ],
        key: str,
    ) -> str:
        """Retrieve a prompt by kind and key.

        Args:
            kind: The kind of prompt.
            key: The key of the specific prompt to retrieve.

        Returns:
            The prompt as a string.

        Raises:
            Exception: If the prompt for the given kind and key is not found.
        """
        try:
            return self._prompts[kind][key]
        except Exception as e:
            raise Exception(f"Prompt for '{kind}':'{key}'  not found.") from e


@lru_cache(maxsize=None)
def get_i18n(prompt_file: str | None = None) -> I18N:
    """Get a cached I18N instance.

    This function caches I18N instances to avoid redundant file I/O and JSON parsing.
    Each unique prompt_file path gets its own cached instance.

    Args:
        prompt_file: Optional custom prompt file path. Defaults to None (uses built-in prompts).

    Returns:
        Cached I18N instance.
    """
    return I18N(prompt_file=prompt_file)
