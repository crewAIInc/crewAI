from __future__ import annotations

import time
from typing import TYPE_CHECKING

from crewai.agents.parser import AgentFinish
from crewai.events.event_listener import event_listener
from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.utilities.converter import ConverterError
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import sanitize_tool_name


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.crew import Crew
    from crewai.task import Task
    from crewai.utilities.i18n import I18N
    from crewai.utilities.types import LLMMessage


class CrewAgentExecutorMixin:
    crew: Crew | None
    agent: Agent
    task: Task | None
    iterations: int
    max_iter: int
    messages: list[LLMMessage]
    _i18n: I18N
    _printer: Printer = Printer()

    def _create_short_term_memory(self, output: AgentFinish) -> None:
        """Create and save a short-term memory item if conditions are met."""
        if (
            self.crew
            and self.agent
            and self.task
            and f"Action: {sanitize_tool_name('Delegate work to coworker')}"
            not in output.text
        ):
            try:
                if (
                    hasattr(self.crew, "_short_term_memory")
                    and self.crew._short_term_memory
                ):
                    self.crew._short_term_memory.save(
                        value=output.text,
                        metadata={
                            "observation": self.task.description,
                        },
                    )
            except Exception as e:
                self.agent._logger.log(
                    "error", f"Failed to add to short term memory: {e}"
                )

    def _create_external_memory(self, output: AgentFinish) -> None:
        """Create and save a external-term memory item if conditions are met."""
        if (
            self.crew
            and self.agent
            and self.task
            and hasattr(self.crew, "_external_memory")
            and self.crew._external_memory
        ):
            try:
                self.crew._external_memory.save(
                    value=output.text,
                    metadata={
                        "description": self.task.description,
                        "messages": self.messages,
                    },
                )
            except Exception as e:
                self.agent._logger.log(
                    "error", f"Failed to add to external memory: {e}"
                )

    def _create_long_term_memory(self, output: AgentFinish) -> None:
        """Create and save long-term and entity memory items based on evaluation."""
        if (
            self.crew
            and self.crew._long_term_memory
            and self.crew._entity_memory
            and self.task
            and self.agent
        ):
            try:
                ltm_agent = TaskEvaluator(self.agent)
                evaluation = ltm_agent.evaluate(self.task, output.text)

                if isinstance(evaluation, ConverterError):
                    return

                long_term_memory = LongTermMemoryItem(
                    task=self.task.description,
                    agent=self.agent.role,
                    quality=evaluation.quality,
                    datetime=str(time.time()),
                    expected_output=self.task.expected_output,
                    metadata={
                        "suggestions": evaluation.suggestions,
                        "quality": evaluation.quality,
                    },
                )
                self.crew._long_term_memory.save(long_term_memory)

                entity_memories = [
                    EntityMemoryItem(
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        relationships="\n".join(
                            [f"- {r}" for r in entity.relationships]
                        ),
                    )
                    for entity in evaluation.entities
                ]
                if entity_memories:
                    self.crew._entity_memory.save(entity_memories)
            except AttributeError as e:
                self.agent._logger.log(
                    "error", f"Missing attributes for long term memory: {e}"
                )
            except Exception as e:
                self.agent._logger.log(
                    "error", f"Failed to add to long term memory: {e}"
                )
        elif (
            self.crew
            and self.crew._long_term_memory
            and self.crew._entity_memory is None
        ):
            if self.agent and self.agent.verbose:
                self._printer.print(
                    content="Long term memory is enabled, but entity memory is not enabled. Please configure entity memory or set memory=True to automatically enable it.",
                    color="bold_yellow",
                )

    def _ask_human_input(self, final_answer: str) -> str:
        """Prompt human input with mode-appropriate messaging.

        Note: The final answer is already displayed via the AgentLogsExecutionEvent
        panel, so we only show the feedback prompt here.
        """
        from rich.panel import Panel
        from rich.text import Text

        formatter = event_listener.formatter
        formatter.pause_live_updates()

        try:
            # Training mode prompt (single iteration)
            if self.crew and getattr(self.crew, "_train", False):
                prompt_text = (
                    "TRAINING MODE: Provide feedback to improve the agent's performance.\n\n"
                    "This will be used to train better versions of the agent.\n"
                    "Please provide detailed feedback about the result quality and reasoning process."
                )
                title = "🎓 Training Feedback Required"
            # Regular human-in-the-loop prompt (multiple iterations)
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
