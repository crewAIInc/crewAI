from __future__ import annotations

from typing import Any, TypedDict

from pydantic import BaseModel, Field

from crewai.utilities.i18n import I18N


class StandardPromptResult(TypedDict):
    """Result with only prompt field for standard mode."""

    prompt: str


class SystemPromptResult(StandardPromptResult):
    """Result with system, user, and prompt fields for system prompt mode."""

    system: str
    user: str


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent."""

    i18n: I18N = Field(default_factory=I18N)
    has_tools: bool = Field(
        default=False, description="Indicates if the agent has access to tools"
    )
    system_template: str | None = Field(
        default=None, description="Custom system prompt template"
    )
    prompt_template: str | None = Field(
        default=None, description="Custom user prompt template"
    )
    response_template: str | None = Field(
        default=None, description="Custom response prompt template"
    )
    use_system_prompt: bool | None = Field(
        default=False,
        description="Whether to use the system prompt when no custom templates are provided",
    )
    agent: Any = Field(description="Reference to the agent using these prompts")

    def task_execution(self) -> SystemPromptResult | StandardPromptResult:
        """Generate a standard prompt for task execution.

        Returns:
            A dictionary containing the constructed prompt(s).
        """
        slices: list[str] = ["role_playing"]
        if self.has_tools:
            slices.append("tools")
        else:
            slices.append("no_tools")
        system: str = self._build_prompt(slices)
        slices.append("task")

        if (
            not self.system_template
            and not self.prompt_template
            and self.use_system_prompt
        ):
            return SystemPromptResult(
                system=system,
                user=self._build_prompt(["task"]),
                prompt=self._build_prompt(slices),
            )
        return StandardPromptResult(
            prompt=self._build_prompt(
                slices,
                self.system_template,
                self.prompt_template,
                self.response_template,
            )
        )

    def _build_prompt(
        self,
        components: list[str],
        system_template: str | None = None,
        prompt_template: str | None = None,
        response_template: str | None = None,
    ) -> str:
        """Constructs a prompt string from specified components.

        Args:
            components: List of component names to include in the prompt.
            system_template: Optional custom template for the system prompt.
            prompt_template: Optional custom template for the user prompt.
            response_template: Optional custom template for the response prompt.

        Returns:
            The constructed prompt string.
        """
        prompt: str
        if not system_template or not prompt_template:
            # If any of the required templates are missing, fall back to the default format
            prompt_parts: list[str] = [
                self.i18n.slice(component) for component in components
            ]
            prompt = "".join(prompt_parts)
        else:
            # All templates are provided, use them
            template_parts: list[str] = [
                self.i18n.slice(component)
                for component in components
                if component != "task"
            ]
            system: str = system_template.replace(
                "{{ .System }}", "".join(template_parts)
            )
            prompt = prompt_template.replace(
                "{{ .Prompt }}", "".join(self.i18n.slice("task"))
            )
            # Handle missing response_template
            if response_template:
                response: str = response_template.split("{{ .Response }}")[0]
                prompt = f"{system}\n{prompt}\n{response}"
            else:
                prompt = f"{system}\n{prompt}"

        return (
            prompt.replace("{goal}", self.agent.goal)
            .replace("{role}", self.agent.role)
            .replace("{backstory}", self.agent.backstory)
        )
