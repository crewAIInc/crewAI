"""Taakoutput representatie en formattering."""

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator

from crewai.tasks.output_format import OutputFormat
from crewai.utilities.types import LLMMessage


class TaskOutput(BaseModel):
    """Klasse die het resultaat van een taak representeert.

    Attributen:
        description: Beschrijving van de taak
        name: Optionele naam van de taak
        expected_output: Verwachte output van de taak
        summary: Samenvatting van de taak (automatisch gegenereerd van beschrijving)
        raw: Ruwe output van de taak
        pydantic: Pydantic model output van de taak
        json_dict: JSON dictionary output van de taak
        agent: Agent die de taak heeft uitgevoerd
        output_format: Output formaat van de taak (JSON, PYDANTIC, of RAW)
        actions_executed: Lijst van uitgevoerde tool acties
        execution_success: Of de taakuitvoering succesvol was
    """

    description: str = Field(description="Beschrijving van de taak")
    name: str | None = Field(description="Naam van de taak", default=None)
    expected_output: str | None = Field(
        description="Verwachte output van de taak", default=None
    )
    summary: str | None = Field(description="Samenvatting van de taak", default=None)
    raw: str = Field(description="Ruwe output van de taak", default="")
    pydantic: BaseModel | None = Field(
        description="Pydantic output van de taak", default=None
    )
    json_dict: dict[str, Any] | None = Field(
        description="JSON dictionary van de taak", default=None
    )
    agent: str = Field(description="Agent die de taak heeft uitgevoerd")
    output_format: OutputFormat = Field(
        description="Output formaat van de taak", default=OutputFormat.RAW
    )
    messages: list[LLMMessage] = Field(description="Berichten van de taak", default=[])
    actions_executed: list[dict[str, Any]] = Field(
        description="Lijst van uitgevoerde tool acties met tool naam, argumenten, resultaat en status",
        default_factory=list
    )
    execution_success: bool = Field(
        description="Of de taakuitvoering succesvol was",
        default=True
    )

    @model_validator(mode="after")
    def set_summary(self):
        """Stel het samenvatting veld in op basis van de beschrijving.

        Retourneert:
            Self met bijgewerkt samenvatting veld.
        """
        excerpt = " ".join(self.description.split(" ")[:10])
        self.summary = f"{excerpt}..."
        return self

    @property
    def json(self) -> str | None:  # type: ignore[override]
        """Haal de JSON string representatie van de taakoutput op.

        Retourneert:
            JSON string representatie van de taakoutput.

        Gooit:
            ValueError: Als output formaat niet JSON is.

        Opmerkingen:
            TODO: Refactor om model_dump_json() te gebruiken om BaseModel methode conflict te vermijden
        """
        if self.output_format != OutputFormat.JSON:
            raise ValueError(
                """
                Ongeldig output formaat gevraagd.
                Als je de JSON output wilt benaderen,
                zorg ervoor dat je de output_json property voor de taak instelt
                """
            )

        return json.dumps(self.json_dict)

    def to_dict(self) -> dict[str, Any]:
        """Converteer json_output en pydantic_output naar een dictionary.

        Retourneert:
            Dictionary representatie van de taakoutput. Geeft prioriteit aan json_dict
            boven pydantic model dump als beide beschikbaar zijn.
        """
        output_dict = {}
        if self.json_dict:
            output_dict.update(self.json_dict)
        elif self.pydantic:
            output_dict.update(self.pydantic.model_dump())
        return output_dict

    def __str__(self) -> str:
        if self.pydantic:
            return str(self.pydantic)
        if self.json_dict:
            return str(self.json_dict)
        if self.actions_executed:
            actions_summary = [
                f"{a.get('tool', 'onbekend')}: {'✓' if a.get('success', False) else '✗'}"
                for a in self.actions_executed
            ]
            return f"Uitgevoerde acties: {', '.join(actions_summary)}"
        return self.raw
