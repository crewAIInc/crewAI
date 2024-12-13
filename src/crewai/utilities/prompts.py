from typing import Any, Optional

from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent."""

    i18n: I18N = Field(default=I18N())
    tools: list[Any] = Field(default=[])
    system_template: Optional[str] = None
    prompt_template: Optional[str] = None
    response_template: Optional[str] = None
    use_system_prompt: Optional[bool] = False
    agent: Any

    def task_execution(self) -> dict[str, str]:
        """Generate a standard prompt for task execution."""
        slices = ["role_playing"]
        if len(self.tools) > 0:
            slices.append("tools")
        else:
            slices.append("no_tools")
        system = self._build_prompt(slices)
        slices.append("task")

        if (
            not self.system_template
            and not self.prompt_template
            and self.use_system_prompt
        ):
            return {
                "system": system,
                "user": self._build_prompt(["task"]),
                "prompt": self._build_prompt(slices),
            }
        else:
            return {
                "prompt": self._build_prompt(
                    slices,
                    self.system_template,
                    self.prompt_template,
                    self.response_template,
                )
            }

    def _build_prompt(
        self,
        components: list[str],
        system_template=None,
        prompt_template=None,
        response_template=None,
    ) -> str:
        """Constructs a prompt string from specified components."""
        if not system_template and not prompt_template:
            prompt_parts = [self.i18n.slice(component) for component in components]
            prompt = "".join(prompt_parts)
        else:
            prompt_parts = [
                self.i18n.slice(component)
                for component in components
                if component != "task"
            ]
            system = system_template.replace("{{ .System }}", "".join(prompt_parts))
            prompt = prompt_template.replace(
                "{{ .Prompt }}", "".join(self.i18n.slice("task"))
            )
            response = response_template.split("{{ .Response }}")[0]
            prompt = f"{system}\n{prompt}\n{response}"

        prompt = (
            prompt.replace("{goal}", self.agent.goal)
            .replace("{role}", self.agent.role)
            .replace("{backstory}", self.agent.backstory)
        )
        return prompt
