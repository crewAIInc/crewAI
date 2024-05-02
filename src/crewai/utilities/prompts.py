from typing import Any, ClassVar, Optional

from langchain.prompts import BasePromptTemplate, PromptTemplate
from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent."""

    i18n: I18N = Field(default=I18N())
    tools: list[Any] = Field(default=[])
    system_template: Optional[str] = None
    prompt_template: Optional[str] = None
    response_template: Optional[str] = None
    SCRATCHPAD_SLICE: ClassVar[str] = "\n{agent_scratchpad}"

    def task_execution(self) -> BasePromptTemplate:
        """Generate a standard prompt for task execution."""
        slices = ["role_playing"]
        if len(self.tools) > 0:
            slices.append("tools")
        else:
            slices.append("no_tools")

        slices.append("task")

        if not self.system_template and not self.prompt_template:
            return self._build_prompt(slices)
        else:
            return self._build_prompt(
                slices,
                self.system_template,
                self.prompt_template,
                self.response_template,
            )

    def _build_prompt(
        self,
        components: list[str],
        system_template=None,
        prompt_template=None,
        response_template=None,
    ) -> BasePromptTemplate:
        """Constructs a prompt string from specified components."""
        if not system_template and not prompt_template:
            prompt_parts = [self.i18n.slice(component) for component in components]
            prompt_parts.append(self.SCRATCHPAD_SLICE)
            prompt = PromptTemplate.from_template("".join(prompt_parts))
        else:
            prompt_parts = [
                self.i18n.slice(component)
                for component in components
                if component != "task"
            ]
            system = system_template.replace("{{ .System }}", "".join(prompt_parts))
            prompt = prompt_template.replace(
                "{{ .Prompt }}",
                "".join([self.i18n.slice("task"), self.SCRATCHPAD_SLICE]),
            )
            response = response_template.split("{{ .Response }}")[0]
            prompt = PromptTemplate.from_template(f"{system}\n{prompt}\n{response}")
        return prompt
