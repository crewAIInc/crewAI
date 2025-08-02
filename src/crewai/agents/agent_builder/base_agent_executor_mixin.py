import time
import uuid
from typing import TYPE_CHECKING

from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.utilities import I18N
from crewai.utilities.converter import ConverterError
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator
from crewai.utilities.printer import Printer
from crewai.utilities.events.event_listener import event_listener
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.events.task_events import HumanInputRequiredEvent, HumanInputCompletedEvent

if TYPE_CHECKING:
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.crew import Crew
    from crewai.task import Task


class CrewAgentExecutorMixin:
    crew: "Crew"
    agent: "BaseAgent"
    task: "Task"
    iterations: int
    max_iter: int
    _i18n: I18N
    _printer: Printer = Printer()

    def _create_short_term_memory(self, output) -> None:
        """Create and save a short-term memory item if conditions are met."""
        if (
            self.crew
            and self.agent
            and self.task
            and "Action: Delegate work to coworker" not in output.text
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
                        agent=self.agent.role,
                    )
            except Exception as e:
                print(f"Failed to add to short term memory: {e}")
                pass

    def _create_external_memory(self, output) -> None:
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
                    },
                    agent=self.agent.role,
                )
            except Exception as e:
                print(f"Failed to add to external memory: {e}")
                pass

    def _create_long_term_memory(self, output) -> None:
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

                for entity in evaluation.entities:
                    entity_memory = EntityMemoryItem(
                        name=entity.name,
                        type=entity.type,
                        description=entity.description,
                        relationships="\n".join(
                            [f"- {r}" for r in entity.relationships]
                        ),
                    )
                    self.crew._entity_memory.save(entity_memory)
            except AttributeError as e:
                print(f"Missing attributes for long term memory: {e}")
                pass
            except Exception as e:
                print(f"Failed to add to long term memory: {e}")
                pass
        elif (
            self.crew
            and self.crew._long_term_memory
            and self.crew._entity_memory is None
        ):
            self._printer.print(
                content="Long term memory is enabled, but entity memory is not enabled. Please configure entity memory or set memory=True to automatically enable it.",
                color="bold_yellow",
            )

    def _ask_human_input(self, final_answer: str) -> str:
        """Prompt human input with mode-appropriate messaging."""
        event_id = str(uuid.uuid4())
        
        execution_id = getattr(self.crew, 'id', None) if self.crew else None
        crew_id = str(execution_id) if execution_id else None
        task_id = str(getattr(self.task, 'id', None)) if self.task else None
        agent_id = str(getattr(self.agent, 'id', None)) if self.agent else None
        
        if self.crew and getattr(self.crew, "_train", False):
            prompt_text = (
                "## TRAINING MODE: Provide feedback to improve the agent's performance.\n"
                "This will be used to train better versions of the agent.\n"
                "Please provide detailed feedback about the result quality and reasoning process."
            )
        else:
            prompt_text = (
                "## HUMAN FEEDBACK: Provide feedback on the Final Result and Agent's actions.\n"
                "Please follow these guidelines:\n"
                " - If you are happy with the result, simply hit Enter without typing anything.\n"
                " - Otherwise, provide specific improvement requests.\n"
                " - You can provide multiple rounds of feedback until satisfied."
            )
        
        crewai_event_bus.emit(
            self,
            HumanInputRequiredEvent(
                execution_id=execution_id,
                crew_id=crew_id,
                task_id=task_id,
                agent_id=agent_id,
                prompt=prompt_text,
                context=final_answer,
                event_id=event_id,
                reason_flags={"ambiguity": True, "missing_field": False}
            )
        )
        
        event_listener.formatter.pause_live_updates()
        
        try:
            self._printer.print(
                content=f"\033[1m\033[95m ## Final Result:\033[00m \033[92m{final_answer}\033[00m"
            )

            if self.crew and getattr(self.crew, "_train", False):
                prompt = (
                    "\n\n=====\n"
                    "## TRAINING MODE: Provide feedback to improve the agent's performance.\n"
                    "This will be used to train better versions of the agent.\n"
                    "Please provide detailed feedback about the result quality and reasoning process.\n"
                    "=====\n"
                )
            else:
                prompt = (
                    "\n\n=====\n"
                    "## HUMAN FEEDBACK: Provide feedback on the Final Result and Agent's actions.\n"
                    "Please follow these guidelines:\n"
                    " - If you are happy with the result, simply hit Enter without typing anything.\n"
                    " - Otherwise, provide specific improvement requests.\n"
                    " - You can provide multiple rounds of feedback until satisfied.\n"
                    "=====\n"
                )

            self._printer.print(content=prompt, color="bold_yellow")
            response = input()
            if response.strip() != "":
                self._printer.print(content="\nProcessing your feedback...", color="cyan")
            
            crewai_event_bus.emit(
                self,
                HumanInputCompletedEvent(
                    execution_id=execution_id,
                    crew_id=crew_id,
                    task_id=task_id,
                    agent_id=agent_id,
                    event_id=event_id,
                    human_feedback=response
                )
            )
            
            return response
        finally:
            event_listener.formatter.resume_live_updates()
