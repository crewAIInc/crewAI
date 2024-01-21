from typing import ClassVar

from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent with support for different languages."""

    i18n: I18N = Field(default=I18N())

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
        prompt_parts = [self.i18n.slice(component) for component in components]
        prompt_parts.append(self.SCRATCHPAD_SLICE)
        return PromptTemplate.from_template("".join(prompt_parts))
