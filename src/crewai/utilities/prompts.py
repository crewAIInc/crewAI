from typing import Any, ClassVar

from langchain.prompts import BasePromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent with support for different languages."""

    i18n: I18N = Field(default=I18N())
    tools: list[Any] = Field(default=[])
    SCRATCHPAD_SLICE: ClassVar[str] = "\n{agent_scratchpad}"

    def task_execution_without_tools(self) -> BasePromptTemplate:
        """Generate a prompt for task execution without tools components."""
        return self._build_prompt(["role_playing", "task"])

    def task_execution(self) -> BasePromptTemplate:
        """Generate a standard prompt for task execution."""
        slices = ["role_playing"]
        if len(self.tools) > 0:
            slices.append("tools")
        else:
            slices.append("no_tools")
        slices.append("task")
        return self._build_prompt(slices)

    def _build_prompt(self, components: list[str]) -> BasePromptTemplate:
        """Constructs a prompt string from specified components."""
        prompt_parts = [self.i18n.slice(component) for component in components]
        prompt_parts.append(self.SCRATCHPAD_SLICE)
        prompt = PromptTemplate.from_template("".join(prompt_parts))
        return prompt
