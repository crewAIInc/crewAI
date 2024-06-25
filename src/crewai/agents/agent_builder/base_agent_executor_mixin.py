import time

from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.utilities.converter import ConverterError
from crewai.utilities.evaluators.task_evaluator import TaskEvaluator


class CrewAgentExecutorMixin:
    def _should_force_answer(self) -> bool:
        return (
            self.iterations == self.force_answer_max_iterations
        ) and not self.have_forced_answer

    def _create_short_term_memory(self, output) -> None:
        if (
            self.crew
            and self.crew.memory
            and "Action: Delegate work to coworker" not in output.log
        ):
            memory = ShortTermMemoryItem(
                data=output.log,
                agent=self.crew_agent.role,
                metadata={
                    "observation": self.task.description,
                },
            )
            self.crew._short_term_memory.save(memory)

    def _create_long_term_memory(self, output) -> None:
        if self.crew and self.crew.memory:
            ltm_agent = TaskEvaluator(self.crew_agent)
            evaluation = ltm_agent.evaluate(self.task, output.log)

            if isinstance(evaluation, ConverterError):
                return

            long_term_memory = LongTermMemoryItem(
                task=self.task.description,
                agent=self.crew_agent.role,
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
                    relationships="\n".join([f"- {r}" for r in entity.relationships]),
                )
                self.crew._entity_memory.save(entity_memory)

    def _ask_human_input(self, final_answer: dict) -> str:
        """Get human input."""
        return input(
            self._i18n.slice("getting_input").format(final_answer=final_answer)
        )
