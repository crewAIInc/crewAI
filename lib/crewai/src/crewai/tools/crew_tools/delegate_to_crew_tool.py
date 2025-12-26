"""Tool voor het delegeren van werk naar andere crews."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.i18n import I18N, get_i18n

if TYPE_CHECKING:
    from crewai.crew import Crew


class DelegateToCrewSchema(BaseModel):
    """Schema voor crew delegatie parameters."""

    crew_name: str = Field(
        ...,
        description="Naam van de crew om naar te delegeren",
    )
    directive: str = Field(
        ...,
        description="De opdracht of taak voor de crew",
    )
    context: str = Field(
        ...,
        description="Alle benodigde context en informatie voor de crew",
    )


class DelegateToCrewTool(BaseTool):
    """Tool voor het delegeren van opdrachten naar andere crews.

    Hiermee kan een manager-crew taken delegeren aan gespecialiseerde crews
    zoals een trading-crew, analyse-crew, of monitoring-crew.

    Voorbeeld:
        ```python
        tool = DelegateToCrewTool(crews={
            "spot": spot_crew,
            "futures": futures_crew,
        })

        # Agent kan nu delegeren:
        # Actie: Delegeer naar Crew
        # Actie Input: {
        #     "crew_name": "spot",
        #     "directive": "Koop 0.01 BTC",
        #     "context": "Marktprijs is $42,000"
        # }
        ```
    """

    name: str = "Delegeer naar Crew"
    description: str = ""
    args_schema: type[BaseModel] = DelegateToCrewSchema
    crews: dict[str, Any] = Field(default_factory=dict)
    i18n: I18N = Field(default_factory=get_i18n)

    def __init__(self, crews: dict[str, Any], **kwargs: Any) -> None:
        """Initialiseer de crew delegatie tool.

        Args:
            crews: Dictionary met crew namen als keys en Crew instances als values.
            **kwargs: Extra argumenten voor BaseTool.
        """
        crew_names = ", ".join(crews.keys())
        description = f"""Delegeer een opdracht naar een van de volgende crews: {crew_names}

De input moet bevatten:
- crew_name: De naam van de crew (één van: {crew_names})
- directive: De specifieke opdracht voor de crew
- context: Alle benodigde context en informatie

Gebruik deze tool om werk te delegeren naar gespecialiseerde teams.
De crew zal de opdracht uitvoeren en het resultaat terugrapporteren."""

        super().__init__(crews=crews, description=description, **kwargs)

    def _run(
        self,
        crew_name: str,
        directive: str,
        context: str,
        **kwargs: Any,
    ) -> str:
        """Voer de delegatie uit naar de gespecificeerde crew.

        Args:
            crew_name: Naam van de target crew.
            directive: De opdracht voor de crew.
            context: Context informatie.
            **kwargs: Extra argumenten (genegeerd).

        Returns:
            Het resultaat van de crew uitvoering of een foutmelding.
        """
        if crew_name not in self.crews:
            available = ", ".join(self.crews.keys())
            return f"Fout: Crew '{crew_name}' niet gevonden. Beschikbare crews: {available}"

        crew = self.crews[crew_name]

        try:
            # Start de crew met de gegeven inputs
            result = crew.kickoff(
                inputs={
                    "directive": directive,
                    "context": context,
                }
            )

            # Haal het resultaat op
            if hasattr(result, "raw"):
                return f"Crew '{crew_name}' heeft de opdracht voltooid:\n\n{result.raw}"
            else:
                return f"Crew '{crew_name}' heeft de opdracht voltooid:\n\n{str(result)}"

        except Exception as e:
            return f"Fout bij delegatie naar crew '{crew_name}': {str(e)}"
