from typing import ClassVar

from langchain.prompts import PromptTemplate, BasePromptTemplate
from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent with support for different languages."""

    i18n: I18N = Field(default=I18N())

    SCRATCHPAD_SLICE: ClassVar[str] = "\n{agent_scratchpad}"

    def task_execution_with_memory(self) -> BasePromptTemplate:
        """        Generate a prompt for task execution with memory components.

        Returns:
            BasePromptTemplate: A prompt template for task execution with memory components.
        """
        return self._build_prompt(["role_playing", "tools", "memory", "task"])

    def task_execution_without_tools(self) -> BasePromptTemplate:
        """        Generate a prompt for task execution without tools components.

        Returns:
            BasePromptTemplate: The generated prompt template.
        """
        return self._build_prompt(["role_playing", "task"])

    def task_execution(self) -> BasePromptTemplate:
        """        Generate a standard prompt for task execution.

        Returns:
            BasePromptTemplate: A prompt template for task execution.
        """
        return self._build_prompt(["role_playing", "tools", "task"])

    def _build_prompt(self, components: list[str]) -> BasePromptTemplate:
        """        Constructs a prompt string from specified components.

        Args:
            components (list[str]): List of components to construct the prompt string.

        Returns:
            BasePromptTemplate: Prompt template constructed from the specified components.
        """
        prompt_parts = [self.i18n.slice(component) for component in components]
        prompt_parts.append(self.SCRATCHPAD_SLICE)
        return PromptTemplate.from_template("".join(prompt_parts))
