"""Tool voor het rapporteren aan management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from crewai.communication.reports import RapportManager
    from crewai.organization.hierarchy import OrganisatieHierarchie


class RapporteerSchema(BaseModel):
    """Schema voor rapport parameters."""

    type: str = Field(
        ...,
        description="Type rapport: status, probleem, resultaat, voortgang, of aanbeveling",
    )
    titel: str = Field(
        ...,
        description="Korte titel van het rapport",
    )
    inhoud: str = Field(
        ...,
        description="Gedetailleerde inhoud van het rapport",
    )
    prioriteit: str = Field(
        default="normaal",
        description="Prioriteit: info, normaal, belangrijk, of urgent",
    )


class RapporteerTool(BaseTool):
    """Tool voor het rapporteren aan management.

    Hiermee kan een agent of crew rapporteren aan hun manager(s)
    over status, problemen, resultaten, etc.

    Voorbeeld:
        ```python
        from crewai.tools.organization_tools import RapporteerTool

        tool = RapporteerTool(
            agent_id=medewerker_id,
            rapport_manager=rapport_mgr,
            ontvangers={"manager": manager_id}
        )

        # Agent kan nu rapporteren:
        # Actie: Rapporteer
        # Actie Input: {
        #     "type": "status",
        #     "titel": "Dagelijkse update",
        #     "inhoud": "Alle taken zijn voltooid"
        # }
        ```
    """

    name: str = "Rapporteer"
    description: str = ""
    args_schema: type[BaseModel] = RapporteerSchema

    # Configuratie
    agent_id: UUID = Field(
        ...,
        description="ID van de agent die deze tool gebruikt",
    )
    organisatie: Any = Field(
        default=None,
        description="Referentie naar OrganisatieHierarchie",
    )
    rapport_manager: Any = Field(
        default=None,
        description="Referentie naar RapportManager",
    )
    ontvangers: dict[str, UUID] = Field(
        default_factory=dict,
        description="Mapping van namen naar IDs van managers",
    )

    def __init__(
        self,
        agent_id: UUID,
        organisatie: "OrganisatieHierarchie | None" = None,
        rapport_manager: "RapportManager | None" = None,
        ontvangers: dict[str, UUID] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialiseer de rapport tool.

        Args:
            agent_id: ID van de agent die deze tool gebruikt.
            organisatie: Optionele organisatie voor permissiecontrole.
            rapport_manager: Optionele rapport manager.
            ontvangers: Mapping van namen naar manager IDs.
            **kwargs: Extra argumenten voor BaseTool.
        """
        ontvangers = ontvangers or {}

        description = """Rapporteer aan je management.

Rapport types:
- status: Statusupdate over lopende werkzaamheden
- probleem: Melding van een probleem of blokkade
- resultaat: Rapportage van behaald resultaat
- voortgang: Periodieke voortgangsrapportage
- aanbeveling: Rapport met aanbevelingen voor actie

De input moet bevatten:
- type: Type rapport (status/probleem/resultaat/voortgang/aanbeveling)
- titel: Korte titel
- inhoud: Gedetailleerde inhoud
- prioriteit: info/normaal/belangrijk/urgent (optioneel, standaard: normaal)

Gebruik deze tool om je manager op de hoogte te houden."""

        super().__init__(
            agent_id=agent_id,
            organisatie=organisatie,
            rapport_manager=rapport_manager,
            ontvangers=ontvangers,
            description=description,
            **kwargs,
        )

    def _run(
        self,
        type: str,
        titel: str,
        inhoud: str,
        prioriteit: str = "normaal",
        **kwargs: Any,
    ) -> str:
        """Stuur een rapport naar management.

        Args:
            type: Type rapport.
            titel: Titel van het rapport.
            inhoud: Inhoud van het rapport.
            prioriteit: Prioriteit niveau.
            **kwargs: Extra argumenten (genegeerd).

        Returns:
            Bevestiging of foutmelding.
        """
        # Map type string
        type_mapping = {
            "status": "status",
            "probleem": "probleem",
            "resultaat": "resultaat",
            "voortgang": "voortgang",
            "aanbeveling": "aanbeveling",
            "analyse": "analyse",
            "escalatie": "escalatie",
        }
        rapport_type = type_mapping.get(type.lower())
        if rapport_type is None:
            return f"Fout: Ongeldig rapport type '{type}'. Gebruik: status, probleem, resultaat, voortgang, of aanbeveling"

        # Map prioriteit
        prioriteit_mapping = {
            "info": "info",
            "normaal": "normaal",
            "belangrijk": "belangrijk",
            "urgent": "urgent",
        }
        prioriteit_waarde = prioriteit_mapping.get(prioriteit.lower(), "normaal")

        # Bepaal ontvangers
        ontvanger_ids = list(self.ontvangers.values())

        # Als geen expliciete ontvangers, probeer uit organisatie
        if not ontvanger_ids and self.organisatie is not None:
            managers = self.organisatie.krijg_managers(self.agent_id)
            ontvanger_ids = managers

        if not ontvanger_ids:
            return "Fout: Geen ontvangers geconfigureerd en geen managers gevonden in organisatie"

        # Stuur rapport via manager indien beschikbaar
        if self.rapport_manager is not None:
            try:
                from crewai.communication.reports import RapportPrioriteit, RapportType

                rapport_type_enum = RapportType(rapport_type)
                prioriteit_enum = RapportPrioriteit(prioriteit_waarde)

                nieuw_rapport = self.rapport_manager.stuur_rapport(
                    van_id=self.agent_id,
                    naar_ids=ontvanger_ids,
                    type=rapport_type_enum,
                    titel=titel,
                    samenvatting=inhoud[:200] + ("..." if len(inhoud) > 200 else ""),
                    details={"volledige_inhoud": inhoud},
                    prioriteit=prioriteit_enum,
                )

                ontvangers_str = ", ".join(
                    [
                        naam
                        for naam, id in self.ontvangers.items()
                        if id in ontvanger_ids
                    ]
                )
                if not ontvangers_str:
                    ontvangers_str = f"{len(ontvanger_ids)} manager(s)"

                return f"""Rapport succesvol verstuurd.

Rapport ID: {nieuw_rapport.id}
Type: {rapport_type}
Titel: {titel}
Prioriteit: {prioriteit_waarde}
Verzonden naar: {ontvangers_str}

Je management is op de hoogte gebracht."""

            except PermissionError as e:
                return f"Fout: {str(e)}"
            except Exception as e:
                return f"Fout bij versturen rapport: {str(e)}"

        # Zonder manager, alleen bevestiging
        return f"""Rapport geregistreerd.

Type: {rapport_type}
Titel: {titel}
Prioriteit: {prioriteit_waarde}

Let op: Geen rapport manager geconfigureerd - rapport wordt niet formeel geregistreerd."""
