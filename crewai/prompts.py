"""Prompts for generic agent."""
import json
from typing import ClassVar, Dict, Optional

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field, PrivateAttr, model_validator


class Prompts(BaseModel):
    """Prompts for generic agent."""

    _prompts: Optional[Dict[str, str]] = PrivateAttr()
    language: Optional[str] = Field(
        default="en",
        description="Language of crewai prompts.",
    )

    @model_validator(mode="after")
    def load_prompts(self) -> "Prompts":
        """Load prompts from file."""
        with open(f"crewai/prompts/{self.language}.json", "r") as f:
            self._prompts = json.load(f)["slices"]
        return self

    SCRATCHPAD_SLICE: ClassVar[str] = "\n{agent_scratchpad}"

    def task_execution_with_memory(self) -> str:
        return PromptTemplate.from_template(
            self._prompts["role_playing"]
            + self._prompts["tools"]
            + self._prompts["memory"]
            + self._prompts["task"]
            + self.SCRATCHPAD_SLICE
        )

    def task_execution_without_tools(self) -> str:
        return PromptTemplate.from_template(
            self._prompts["role_playing"]
            + self._prompts["task"]
            + self.SCRATCHPAD_SLICE
        )

    def task_execution(self) -> str:
        return PromptTemplate.from_template(
            self._prompts["role_playing"]
            + self._prompts["tools"]
            + self._prompts["task"]
            + self.SCRATCHPAD_SLICE
        )
