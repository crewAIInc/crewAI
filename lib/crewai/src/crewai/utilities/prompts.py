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


COMPONENTS = Literal["role_playing", "tools", "no_tools", "task"]


class Prompts(BaseModel):
    """Manages and generates prompts for a generic agent.

    Notes:
        - Need to refactor so that prompt is not tightly coupled to agent.
    """

    i18n: I18N = Field(default_factory=get_i18n)
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

    def continuous_execution(self) -> SystemPromptResult:
        """Generate prompts for continuous operation mode.

        Continuous mode prompts instruct the agent to operate indefinitely,
        monitoring conditions and taking action as needed without providing
        a "Final Answer".

        Returns:
            A SystemPromptResult with system and user prompts for continuous mode.
        """
        system_prompt = self._build_continuous_system_prompt()
        user_prompt = self._build_continuous_user_prompt()

        return SystemPromptResult(
            system=system_prompt,
            user=user_prompt,
            prompt=f"{system_prompt}\n\n{user_prompt}",
        )

    def _build_continuous_system_prompt(self) -> str:
        """Build system prompt for continuous mode.

        Returns:
            System prompt string for continuous operation.
        """
        tools_section = ""
        if self.has_tools:
            tools_section = self.i18n.slice("tools")
        else:
            tools_section = "You have no tools available. You can only observe and think."

        return f"""You are {self.agent.role} operating in CONTINUOUS MONITORING MODE.

Your Goal: {self.agent.goal}

Background: {self.agent.backstory}

{tools_section}

CONTINUOUS OPERATION INSTRUCTIONS:
1. You are running continuously - you should NEVER provide a "Final Answer"
2. Monitor conditions and take action when needed using your available tools
3. Report observations and actions clearly
4. If nothing requires immediate action, state what you are observing
5. Always be ready to respond to changing conditions
6. Keep track of what you have done and observed

When you observe something or want to take action, respond in this format:
Thought: [Your reasoning about what you observe or want to do]
Action: [The tool you want to use, or "Observe" if just monitoring]
Action Input: [The input for the tool]

After each observation or action result, continue monitoring and decide on your next action."""

    def _build_continuous_user_prompt(self) -> str:
        """Build user prompt for continuous mode.

        Returns:
            User prompt string for continuous operation.
        """
        return """Begin continuous operation. Monitor conditions and take appropriate actions.
Remember: You are operating continuously. Do not try to finish or provide a final answer.
Instead, observe, act when needed, and continue monitoring.

Current directive: {task}

Start by assessing the current situation and determining if any action is needed."""
