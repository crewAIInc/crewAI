"""Human input provider for HITL (Human-in-the-Loop) flows."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING, Protocol, runtime_checkable


if TYPE_CHECKING:
    from crewai.agent.core import Agent
    from crewai.agents.parser import AgentFinish
    from crewai.crew import Crew
    from crewai.llms.base_llm import BaseLLM
    from crewai.task import Task
    from crewai.utilities.types import LLMMessage


class ExecutorContext(Protocol):
    """Context interface for human input providers to interact with executor."""

    task: Task | None
    crew: Crew | None
    messages: list[LLMMessage]
    ask_for_human_input: bool
    llm: BaseLLM
    agent: Agent

    def _invoke_loop(self) -> AgentFinish:
        """Invoke the agent loop and return the result."""
        ...

    def _is_training_mode(self) -> bool:
        """Check if training mode is active."""
        ...

    def _handle_crew_training_output(
        self,
        result: AgentFinish,
        human_feedback: str | None = None,
    ) -> None:
        """Handle training output."""
        ...

    def _format_feedback_message(self, feedback: str) -> LLMMessage:
        """Format feedback as a message."""
        ...


@runtime_checkable
class HumanInputProvider(Protocol):
    """Protocol for human input handling.

    Implementations handle the full feedback flow:
    - Sync: prompt user, loop until satisfied
    - Async: raise exception for external handling
    """

    def setup_messages(self, context: ExecutorContext) -> bool:
        """Set up messages for execution.

        Called before standard message setup. Allows providers to handle
        conversation resumption or other custom message initialization.

        Args:
            context: Executor context with messages list to modify.

        Returns:
            True if messages were set up (skip standard setup),
            False to use standard setup.
        """
        ...

    def post_setup_messages(self, context: ExecutorContext) -> None:
        """Called after standard message setup.

        Allows providers to modify messages after standard setup completes.
        Only called when setup_messages returned False.

        Args:
            context: Executor context with messages list to modify.
        """
        ...

    def handle_feedback(
        self,
        formatted_answer: AgentFinish,
        context: ExecutorContext,
    ) -> AgentFinish:
        """Handle the full human feedback flow.

        Args:
            formatted_answer: The agent's current answer.
            context: Executor context for callbacks.

        Returns:
            The final answer after feedback processing.

        Raises:
            Exception: Async implementations may raise to signal external handling.
        """
        ...

    @staticmethod
    def _get_output_string(answer: AgentFinish) -> str:
        """Extract output string from answer.

        Args:
            answer: The agent's finished answer.

        Returns:
            String representation of the output.
        """
        if isinstance(answer.output, str):
            return answer.output
        return answer.output.model_dump_json()


class SyncHumanInputProvider(HumanInputProvider):
    """Default synchronous human input via terminal."""

    def setup_messages(self, context: ExecutorContext) -> bool:
        """Use standard message setup.

        Args:
            context: Executor context (unused).

        Returns:
            False to use standard setup.
        """
        return False

    def post_setup_messages(self, context: ExecutorContext) -> None:
        """No-op for sync provider.

        Args:
            context: Executor context (unused).
        """

    def handle_feedback(
        self,
        formatted_answer: AgentFinish,
        context: ExecutorContext,
    ) -> AgentFinish:
        """Handle feedback synchronously with terminal prompts.

        Args:
            formatted_answer: The agent's current answer.
            context: Executor context for callbacks.

        Returns:
            The final answer after feedback processing.
        """
        feedback = self._prompt_input(context.crew)

        if context._is_training_mode():
            return self._handle_training_feedback(formatted_answer, feedback, context)

        return self._handle_regular_feedback(formatted_answer, feedback, context)

    @staticmethod
    def _handle_training_feedback(
        initial_answer: AgentFinish,
        feedback: str,
        context: ExecutorContext,
    ) -> AgentFinish:
        """Process training feedback (single iteration).

        Args:
            initial_answer: The agent's initial answer.
            feedback: Human feedback string.
            context: Executor context for callbacks.

        Returns:
            Improved answer after processing feedback.
        """
        context._handle_crew_training_output(initial_answer, feedback)
        context.messages.append(context._format_feedback_message(feedback))
        improved_answer = context._invoke_loop()
        context._handle_crew_training_output(improved_answer)
        context.ask_for_human_input = False
        return improved_answer

    def _handle_regular_feedback(
        self,
        current_answer: AgentFinish,
        initial_feedback: str,
        context: ExecutorContext,
    ) -> AgentFinish:
        """Process regular feedback with iteration loop.

        Args:
            current_answer: The agent's current answer.
            initial_feedback: Initial human feedback string.
            context: Executor context for callbacks.

        Returns:
            Final answer after all feedback iterations.
        """
        feedback = initial_feedback
        answer = current_answer

        while context.ask_for_human_input:
            if feedback.strip() == "":
                context.ask_for_human_input = False
            else:
                context.messages.append(context._format_feedback_message(feedback))
                answer = context._invoke_loop()
                feedback = self._prompt_input(context.crew)

        return answer

    @staticmethod
    def _prompt_input(crew: Crew | None) -> str:
        """Show rich panel and prompt for input.

        Args:
            crew: The crew instance for context.

        Returns:
            User input string from terminal.
        """
        from rich.panel import Panel
        from rich.text import Text

        from crewai.events.event_listener import event_listener

        formatter = event_listener.formatter
        formatter.pause_live_updates()

        try:
            if crew and getattr(crew, "_train", False):
                prompt_text = (
                    "TRAINING MODE: Provide feedback to improve the agent's performance.\n\n"
                    "This will be used to train better versions of the agent.\n"
                    "Please provide detailed feedback about the result quality and reasoning process."
                )
                title = "🎓 Training Feedback Required"
            else:
                prompt_text = (
                    "Provide feedback on the Final Result above.\n\n"
                    "• If you are happy with the result, simply hit Enter without typing anything.\n"
                    "• Otherwise, provide specific improvement requests.\n"
                    "• You can provide multiple rounds of feedback until satisfied."
                )
                title = "💬 Human Feedback Required"

            content = Text()
            content.append(prompt_text, style="yellow")

            prompt_panel = Panel(
                content,
                title=title,
                border_style="yellow",
                padding=(1, 2),
            )
            formatter.console.print(prompt_panel)

            response = input()
            if response.strip() != "":
                formatter.console.print("\n[cyan]Processing your feedback...[/cyan]")
            return response
        finally:
            formatter.resume_live_updates()


_provider: ContextVar[HumanInputProvider | None] = ContextVar(
    "human_input_provider",
    default=None,
)


def get_provider() -> HumanInputProvider:
    """Get the current human input provider.

    Returns:
        The current provider, or a new SyncHumanInputProvider if none set.
    """
    provider = _provider.get()
    if provider is None:
        initialized_provider = SyncHumanInputProvider()
        set_provider(initialized_provider)
        return initialized_provider
    return provider


def set_provider(provider: HumanInputProvider) -> Token[HumanInputProvider | None]:
    """Set the human input provider for the current context.

    Args:
        provider: The provider to use.

    Returns:
        Token that can be used to reset to previous value.
    """
    return _provider.set(provider)


def reset_provider(token: Token[HumanInputProvider | None]) -> None:
    """Reset the provider to its previous value.

    Args:
        token: Token returned from set_provider.
    """
    _provider.reset(token)
