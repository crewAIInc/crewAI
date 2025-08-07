from typing import Any, Optional

from pydantic import BaseModel, Field

from crewai.utilities import I18N


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent."""

    i18n: I18N = Field(default=I18N())
    has_tools: bool = False
    system_template: Optional[str] = None
    prompt_template: Optional[str] = None
    response_template: Optional[str] = None
    use_system_prompt: Optional[bool] = False
    agent: Any

    def task_execution(self) -> dict[str, str]:
        """Generate a standard prompt for task execution."""
        slices = ["role_playing"]
        if self.has_tools:
            slices.append("tools")
        else:
            slices.append("no_tools")
        system = self._build_prompt(slices)
        slices.append("task")

        if self.use_system_prompt:
            if self.system_template and self.prompt_template:
                system_content = self._build_custom_system(slices[:-1]) # exclude task
                user_content = self._build_custom_user(["task"])
            else:
                system_content = system
                user_content = self._build_prompt(["task"])
            
            return {
                "system": system_content,
                "user": user_content,
                "prompt": self._build_prompt(
                    slices,
                    self.system_template,
                    self.prompt_template,
                    self.response_template,
                )
            }
        else:
            # When not using system prompt, return the merged format
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
        if not system_template or not prompt_template:
            # If any of the required templates are missing, fall back to the default format
            prompt_parts = [self.i18n.slice(component) for component in components]
            prompt = "".join(prompt_parts)
        else:
            # All templates are provided, use them
            prompt_parts = [
                self.i18n.slice(component)
                for component in components
                if component != "task"
            ]
            system = system_template.replace("{{ .System }}", "".join(prompt_parts))
            prompt = prompt_template.replace(
                "{{ .Prompt }}", "".join(self.i18n.slice("task"))
            )
            # Handle missing response_template
            if response_template:
                response = response_template.split("{{ .Response }}")[0]
                prompt = f"{system}\n{prompt}\n{response}"
            else:
                prompt = f"{system}\n{prompt}"

        return self._apply_agent_variables(prompt)

    def _build_custom_system(self, components: list[str]) -> str:
        if self.system_template:
            prompt_parts = [self.i18n.slice(component) for component in components]
            system = self.system_template.replace("{{ .System }}", "".join(prompt_parts))
            return self._apply_agent_variables(system)
        return self._build_prompt(components)

    def _build_custom_user(self, components: list[str]) -> str:
        if self.prompt_template:
            task_content = "".join([self.i18n.slice(component) for component in components])
            user = self.prompt_template.replace("{{ .Prompt }}", task_content)
            return self._apply_agent_variables(user)
        return self._build_prompt(components)

    def _apply_agent_variables(self, text: str) -> str:
        return (
            text.replace("{goal}", self.agent.goal if self.agent else "")
            .replace("{role}", self.agent.role if self.agent else "")
            .replace("{backstory}", self.agent.backstory if self.agent else "")
        )
