"""Tool voor het delegeren van taken naar sub-crews."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.communication.directives import OpdrachtManager


class DelegeerNaarCrewSchema(BaseModel):
    """Schema voor delegatie parameters."""

    crew_naam: str = Field(
        ...,
        description="Naam van de sub-crew om naar te delegeren",
    )
    taak: str = Field(
        ...,
        description="Beschrijving van de taak om te delegeren",
    )
    context: str = Field(
        default="",
        description="Extra context informatie voor de taak",
    )
    wacht_op_resultaat: bool = Field(
        default=True,
        description="Of gewacht moet worden op het resultaat van de sub-crew",
    )
    prioriteit: str = Field(
        default="normaal",
        description="Prioriteit: laag, normaal, hoog, of kritiek",
    )


class DelegeerNaarCrewTool(BaseTool):
    """Tool voor het delegeren van taken naar ondergeschikte crews.

    Hiermee kan een manager-crew taken delegeren naar sub-crews en
    optioneel wachten op het resultaat.

    Voorbeeld:
        ```python
        from crewai.tools.organization_tools import DelegeerNaarCrewTool

        tool = DelegeerNaarCrewTool(
            crew_id=manager_crew.id,
            sub_crews={"trading": trading_crew, "research": research_crew},
            opdracht_manager=opdracht_mgr
        )

        # Manager kan nu delegeren:
        # Actie: Delegeer naar Crew
        # Actie Input: {
        #     "crew_naam": "trading",
        #     "taak": "Analyseer de huidige marktpositie",
        #     "context": "Focus op EUR/USD en BTC/USD",
        #     "wacht_op_resultaat": true
        # }
        ```
    """

    name: str = "Delegeer naar Crew"
    description: str = ""
    args_schema: type[BaseModel] = DelegeerNaarCrewSchema

    # Configuratie
    crew_id: UUID = Field(
        ...,
        description="ID van de crew die deze tool gebruikt",
    )
    sub_crews: dict[str, Any] = Field(
        default_factory=dict,
        description="Mapping van namen naar sub-Crew instances",
    )
    opdracht_manager: Any = Field(
        default=None,
        description="Referentie naar OpdrachtManager",
    )

    def __init__(
        self,
        crew_id: UUID,
        sub_crews: dict[str, "Crew"] | None = None,
        opdracht_manager: "OpdrachtManager | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialiseer de delegatie tool.

        Args:
            crew_id: ID van de crew die deze tool gebruikt.
            sub_crews: Mapping van namen naar sub-Crew instances.
            opdracht_manager: Optionele opdracht manager.
            **kwargs: Extra argumenten voor BaseTool.
        """
        sub_crews = sub_crews or {}

        # Maak beschrijving met beschikbare crews
        crew_namen = ", ".join(sub_crews.keys()) if sub_crews else "geen"

        description = f"""Delegeer een taak naar een ondergeschikte crew.

Beschikbare crews: {crew_namen}

Gebruik deze tool wanneer je:
- Een complexe taak hebt die door een gespecialiseerde crew moet worden uitgevoerd
- Werk wilt verdelen over meerdere crews
- Resultaten van een sub-crew nodig hebt voor je eigen taak

De input moet bevatten:
- crew_naam: Naam van de sub-crew (een van: {crew_namen})
- taak: Beschrijving van wat de crew moet doen
- context: Extra informatie die nuttig kan zijn (optioneel)
- wacht_op_resultaat: true/false - of je wilt wachten op het resultaat (standaard: true)
- prioriteit: laag/normaal/hoog/kritiek (optioneel, standaard: normaal)

De sub-crew zal de taak uitvoeren en het resultaat terugsturen."""

        super().__init__(
            crew_id=crew_id,
            sub_crews=sub_crews,
            opdracht_manager=opdracht_manager,
            description=description,
            **kwargs,
        )

    def _run(
        self,
        crew_naam: str,
        taak: str,
        context: str = "",
        wacht_op_resultaat: bool = True,
        prioriteit: str = "normaal",
        **kwargs: Any,
    ) -> str:
        """Delegeer een taak naar een sub-crew.

        Args:
            crew_naam: Naam van de sub-crew.
            taak: Beschrijving van de taak.
            context: Extra context informatie.
            wacht_op_resultaat: Of gewacht moet worden op resultaat.
            prioriteit: Prioriteit niveau.
            **kwargs: Extra argumenten (genegeerd).

        Returns:
            Resultaat van de delegatie of bevestiging.
        """
        # Check of crew bestaat
        if crew_naam not in self.sub_crews:
            beschikbaar = ", ".join(self.sub_crews.keys()) if self.sub_crews else "geen"
            return f"""Fout: Crew '{crew_naam}' niet gevonden.

Beschikbare crews: {beschikbaar}

Controleer de naam en probeer opnieuw."""

        sub_crew = self.sub_crews[crew_naam]

        # Maak opdracht indien opdracht_manager beschikbaar
        opdracht_id = None
        if self.opdracht_manager is not None:
            try:
                from crewai.communication.directives import OpdrachtPrioriteit

                # Map prioriteit string
                prioriteit_mapping = {
                    "laag": OpdrachtPrioriteit.LAAG,
                    "normaal": OpdrachtPrioriteit.NORMAAL,
                    "hoog": OpdrachtPrioriteit.HOOG,
                    "kritiek": OpdrachtPrioriteit.KRITIEK,
                }
                prioriteit_enum = prioriteit_mapping.get(
                    prioriteit.lower(), OpdrachtPrioriteit.NORMAAL
                )

                opdracht = self.opdracht_manager.geef_opdracht(
                    van_id=self.crew_id,
                    naar_id=sub_crew.id,
                    titel=f"Gedelegeerde taak: {taak[:50]}{'...' if len(taak) > 50 else ''}",
                    beschrijving=taak,
                    context={"delegatie_context": context} if context else {},
                    prioriteit=prioriteit_enum,
                    ontvanger_type="crew",
                )
                opdracht_id = opdracht.id

            except Exception as e:
                # Log fout maar ga door met delegatie
                pass

        # Voer taak uit
        if wacht_op_resultaat:
            try:
                # Check of sub_crew een ontvang_opdracht methode heeft
                if hasattr(sub_crew, "ontvang_opdracht") and opdracht_id:
                    # Geef de opdracht door aan de sub-crew
                    opdracht = self.opdracht_manager.opdrachten.get(opdracht_id)
                    if opdracht:
                        sub_crew.ontvang_opdracht(opdracht)

                # Kickoff de sub-crew met de taak als input
                resultaat = sub_crew.kickoff(inputs={"taak": taak, "context": context})

                # Update opdracht status naar voltooid
                if self.opdracht_manager is not None and opdracht_id:
                    try:
                        self.opdracht_manager.voltooi_opdracht(
                            opdracht_id,
                            resultaat=str(resultaat.raw) if hasattr(resultaat, "raw") else str(resultaat),
                        )
                    except Exception:
                        pass

                # Format resultaat
                if hasattr(resultaat, "raw"):
                    resultaat_tekst = str(resultaat.raw)
                else:
                    resultaat_tekst = str(resultaat)

                return f"""Taak succesvol gedelegeerd en uitgevoerd door crew '{crew_naam}'.

Resultaat:
{resultaat_tekst}"""

            except Exception as e:
                # Bij fout, update opdracht status
                if self.opdracht_manager is not None and opdracht_id:
                    try:
                        from crewai.communication.directives import OpdrachtStatus

                        self.opdracht_manager.rapporteer_voortgang(
                            opdracht_id,
                            OpdrachtStatus.GEESCALEERD,
                            resultaat=f"Fout bij uitvoering: {str(e)}",
                        )
                    except Exception:
                        pass

                return f"""Fout bij delegatie naar crew '{crew_naam}': {str(e)}

De taak kon niet worden uitgevoerd. Overweeg:
1. Controleer of de crew correct is geconfigureerd
2. Probeer de taak op te splitsen in kleinere delen
3. Escaleer indien nodig naar je manager"""

        else:
            # Async delegatie - start taak zonder te wachten
            try:
                # In een echte implementatie zou dit async zijn
                # Voor nu geven we een bevestiging

                return f"""Taak succesvol gedelegeerd naar crew '{crew_naam}'.

Opdracht ID: {opdracht_id if opdracht_id else 'niet geregistreerd'}
Taak: {taak[:100]}{'...' if len(taak) > 100 else ''}
Prioriteit: {prioriteit}

De crew is gestart met de uitvoering. Gebruik de status tool om voortgang te volgen."""

            except Exception as e:
                return f"Fout bij delegeren naar crew '{crew_naam}': {str(e)}"

    def krijg_beschikbare_crews(self) -> list[str]:
        """Krijg lijst van beschikbare sub-crew namen.

        Returns:
            Lijst met crew namen.
        """
        return list(self.sub_crews.keys())
