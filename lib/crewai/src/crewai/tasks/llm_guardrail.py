from typing import Any

from pydantic import BaseModel, Field

from crewai.agent import Agent
from crewai.lite_agent_output import LiteAgentOutput
from crewai.llms.base_llm import BaseLLM
from crewai.tasks.task_output import TaskOutput


class LLMGuardrailResult(BaseModel):
    valid: bool = Field(
        description="Of de taakoutput voldoet aan de guardrail"
    )
    feedback: str | None = Field(
        description="Een feedback over de taakoutput als deze niet geldig is",
        default=None,
    )


class LLMGuardrail:
    """Het valideert de output van een andere taak met behulp van een LLM.

    Deze klasse wordt gebruikt om de output van een Task te valideren op basis van gespecificeerde criteria.
    Het gebruikt een LLM om de output te valideren en geeft feedback als de output niet geldig is.

    Args:
        description (str): De beschrijving van de validatie criteria.
        llm (LLM, optional): Het taalmodel om te gebruiken voor code generatie.
    """

    def __init__(
        self,
        description: str,
        llm: BaseLLM,
    ):
        self.description = description

        self.llm: BaseLLM = llm

    def _validate_output(self, task_output: TaskOutput) -> LiteAgentOutput:
        agent = Agent(
            role="Guardrail Agent",
            goal="Valideer de output van de taak",
            backstory="Je bent een expert in het valideren van de output van een taak. Door effectieve feedback te geven als de output niet geldig is.",
            llm=self.llm,
        )

        query = f"""
        Zorg ervoor dat het volgende taakresultaat voldoet aan de opgegeven guardrail.

        Taakresultaat:
        {task_output.raw}

        Guardrail:
        {self.description}

        Jouw taak:
        - Bevestig of het Taakresultaat voldoet aan de guardrail.
        - Zo niet, geef duidelijke feedback die uitlegt wat er mis is (bijv. met hoeveel het de regel schendt, of welk specifiek onderdeel faalt).
        - Focus alleen op het identificeren van problemen — stel geen correcties voor.
        - Als het Taakresultaat voldoet aan de guardrail, zeg dat het geldig is
        """

        return agent.kickoff(query, response_format=LLMGuardrailResult)

    def __call__(self, task_output: TaskOutput) -> tuple[bool, Any]:
        """Valideert de output van een taak op basis van gespecificeerde criteria.

        Args:
            task_output (TaskOutput): De output om te valideren.

        Retourneert:
            Tuple[bool, Any]: Een tuple met:
                - bool: True als validatie geslaagd, anders False
                - Any: Het validatie resultaat of foutmelding
        """

        try:
            result = self._validate_output(task_output)
            if not isinstance(result.pydantic, LLMGuardrailResult):
                raise ValueError("Het guardrail resultaat is geen geldig pydantic model")

            if result.pydantic.valid:
                return True, task_output.raw
            return False, result.pydantic.feedback
        except Exception as e:
            return False, f"Fout bij het valideren van de taakoutput: {e!s}"
