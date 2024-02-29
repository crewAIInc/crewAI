from typing import Any, ClassVar

from langchain.prompts import BasePromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """This class is responsible for managing and generating prompts for a generic agent. 
    It supports different languages through the use of the I18N class."""

    i18n: I18N = Field(default=I18N())  # An instance of the I18N class for handling internationalization
    tools: list[Any] = Field(default=[])  # A list of tools that the agent can use
    SCRATCHPAD_SLICE: ClassVar[str] = "\n{agent_scratchpad}"  # A class variable for the scratchpad slice

    def task_execution_with_memory(self) -> BasePromptTemplate:
        """Generates a prompt for task execution with memory components.

        It first creates a list of slices, starting with 'role_playing'. If there are any tools, it adds 'tools' to the slices,
        otherwise it adds 'no_tools'. It then adds 'memory' and 'task' to the slices. Finally, it builds and returns the prompt.
        """
        slices = ["role_playing"]
        if len(self.tools) > 0:
            slices.append("tools")
        else:
            slices.append("no_tools")
        slices.extend(["memory", "task"])
        return self._build_prompt(slices)

    def task_execution_without_tools(self) -> BasePromptTemplate:
        """Generates a prompt for task execution without tools components.

        It builds and returns a prompt with the slices 'role_playing' and 'task'.
        """
        return self._build_prompt(["role_playing", "task"])

    def task_execution(self) -> BasePromptTemplate:
        """Generates a standard prompt for task execution.

        It first creates a list of slices, starting with 'role_playing'. If there are any tools, it adds 'tools' to the slices,
        otherwise it adds 'no_tools'. It then adds 'task' to the slices. Finally, it builds and returns the prompt.
        """
        slices = ["role_playing"]
        if len(self.tools) > 0:
            slices.append("tools")
        else:
            slices.append("no_tools")
        slices.append("task")
        return self._build_prompt(slices)

    def _build_prompt(self, components: list[str]) -> BasePromptTemplate:
        """Constructs a prompt string from the specified components.

        It retrieves the translation for each component from the i18n instance and appends it to the prompt parts.
        It then appends the SCRATCHPAD_SLICE to the prompt parts. Finally, it joins the prompt parts into a single string
        and creates a PromptTemplate from it.
        """
        prompt_parts = [self.i18n.slice(component) for component in components]
        prompt_parts.append(self.SCRATCHPAD_SLICE)
        prompt = PromptTemplate.from_template("".join(prompt_parts))
        return prompt
