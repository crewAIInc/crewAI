"""Conditionele taakuitvoering gebaseerd op vorige taakoutput."""

from collections.abc import Callable
from typing import Any

from pydantic import Field

from crewai.task import Task
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


class ConditionalTask(Task):
    """Een taak die conditioneel kan worden uitgevoerd gebaseerd op de output van een andere taak.

    Dit taaktype maakt dynamische workflow uitvoering mogelijk gebaseerd op de resultaten van
    vorige taken in de crew uitvoeringsketen.

    Attributen:
        condition: Functie die vorige taakoutput evalueert om uitvoering te bepalen.

    Opmerkingen:
        - Kan niet de enige taak in je crew zijn
        - Kan niet de eerste taak zijn omdat het context nodig heeft van de vorige taak
    """

    condition: Callable[[TaskOutput], bool] | None = Field(
        default=None,
        description="Functie die bepaalt of de taak moet worden uitgevoerd gebaseerd op vorige taakoutput.",
    )

    def __init__(
        self,
        condition: Callable[[Any], bool] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition = condition

    def should_execute(self, context: TaskOutput) -> bool:
        """Bepaalt of de conditionele taak moet worden uitgevoerd gebaseerd op de opgegeven context.

        Args:
            context: De output van de vorige taak die door de conditie zal worden geëvalueerd.

        Retourneert:
            True als de taak moet worden uitgevoerd, anders False.

        Gooit:
            ValueError: Als geen conditie functie is ingesteld.
        """
        if self.condition is None:
            raise ValueError("Geen conditie functie ingesteld voor conditionele taak")
        return self.condition(context)

    def get_skipped_task_output(self) -> TaskOutput:
        """Genereer een TaskOutput voor wanneer de conditionele taak wordt overgeslagen.

        Retourneert:
            Lege TaskOutput met RAW formaat die aangeeft dat de taak werd overgeslagen.
        """
        return TaskOutput(
            description=self.description,
            raw="",
            agent=self.agent.role if self.agent else "",
            output_format=OutputFormat.RAW,
        )
