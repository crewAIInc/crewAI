import json
import os
from typing import ClassVar, Dict, Optional

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, PrivateAttr, model_validator, ValidationError

class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent with support for different languages."""

    _prompts: Optional[Dict[str, str]] = PrivateAttr()
    language: Optional[str] = Field(
        default="en",
        description="Language of the prompts.",
    )

    @model_validator(mode="after")
    def load_prompts(self) -> "Prompts":
        """Load prompts from a JSON file based on the specified language."""
        try:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            prompts_path = os.path.join(dir_path, f"prompts/{self.language}.json")

            with open(prompts_path, "r") as f:
                self._prompts = json.load(f)["slices"]
        except FileNotFoundError:
            raise ValidationError(f"Prompt file for language '{self.language}' not found.")
        except json.JSONDecodeError:
            raise ValidationError(f"Error decoding JSON from the prompts file.")
        return self

    SCRATCHPAD_SLICE: ClassVar[str] = "\n{agent_scratchpad}"

    def task_execution_with_memory(self) -> str:
        """Generate a prompt for task execution with memory components."""
        return self._build_prompt(["role_playing", "tools", "memory", "task"])

    def task_execution_without_tools(self) -> str:
        """Generate a prompt for task execution without tools components."""
        return self._build_prompt(["role_playing", "task"])

    def task_execution(self) -> str:
        """Generate a standard prompt for task execution."""
        return self._build_prompt(["role_playing", "tools", "task"])

    def _build_prompt(self, components: [str]) -> str:
        """Constructs a prompt string from specified components."""
        prompt_parts = [self._prompts[component] for component in components if component in self._prompts]
        prompt_parts.append(self.SCRATCHPAD_SLICE)
        return PromptTemplate.from_template("".join(prompt_parts))
