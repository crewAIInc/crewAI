"""Prompt generation and management utilities for CrewAI agents."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from crewai.utilities.i18n import I18N, get_i18n


class StandardPromptResult(TypedDict):
    """Result with only prompt field for standard mode."""

    prompt: Annotated[str, "The generated prompt string"]


class SystemPromptResult(StandardPromptResult):
    """Result with system, user, and prompt fields for system prompt mode."""

    system: Annotated[str, "The system prompt component"]
    user: Annotated[str, "The user prompt component"]


COMPONENTS = Literal[
    "role_playing",
    "tools",
    "no_tools",
    "native_tools",
    "task",
    "native_task",
    "task_no_tools",
]


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent.

    Notes:
        - Need to refactor so that prompt is not tightly coupled to agent.
    """

    i18n: I18N = Field(default_factory=get_i18n)
    has_tools: bool = Field(
        default=False, description="Indicates if the agent has access to tools"
    )
    use_native_tool_calling: bool = Field(
        default=False,
        description="Whether to use native function calling instead of ReAct format",
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
    use_system_prompt: bool = Field(
        default=False,
        description="Whether to use the system prompt when no custom templates are provided",
    )
    agent: Any = Field(description="Reference to the agent using these prompts")

    def task_execution(self) -> SystemPromptResult | StandardPromptResult:
        """Generate a standard prompt for task execution.

        Returns:
            A dictionary containing the constructed prompt(s).
        """
        slices: list[COMPONENTS] = ["role_playing"]
        # When using native tool calling with tools, use native_tools instructions
        # When using ReAct pattern with tools, use tools instructions
        # When no tools are available, use no_tools instructions
        if self.has_tools:
            if not self.use_native_tool_calling:
                slices.append("tools")
        else:
            slices.append("no_tools")
        system: str = self._build_prompt(slices)

        # Determine which task slice to use:
        task_slice: COMPONENTS
        if self.use_native_tool_calling:
            task_slice = "native_task"
        elif self.has_tools:
            task_slice = "task"
        else:
            task_slice = "task_no_tools"
        slices.append(task_slice)

        if (
            not self.system_template
            and not self.prompt_template
            and self.use_system_prompt
        ):
            return SystemPromptResult(
                system=system,
                user=self._build_prompt([task_slice]),
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
        components: list[COMPONENTS],
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
