"""Tool voor escalatie naar hoger management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from crewai.governance.escalation import EscalatieManager
    from crewai.organization.hierarchy import OrganisatieHierarchie


class EscaleerSchema(BaseModel):
    """Schema voor escalatie parameters."""

    reden: str = Field(
        ...,
        description="Reden voor de escalatie",
    )
    context: str = Field(
        ...,
        description="Alle relevante context en informatie",
    )
    urgentie: str = Field(
        default="normaal",
        description="Urgentie: laag, normaal, hoog, of kritiek",
    )


class EscaleerTool(BaseTool):
    """Tool voor het escaleren van problemen naar hoger management.

    Hiermee kan een agent problemen escaleren die ze niet zelf
    kunnen oplossen naar hun manager of hoger.

    Voorbeeld:
        ```python
        from crewai.tools.organization_tools import EscaleerTool

        tool = EscaleerTool(
            agent_id=medewerker_id,
            organisatie=org,
            escalatie_manager=escalatie_mgr
        )

        # Agent kan nu escaleren:
        # Actie: Escaleer
        # Actie Input: {
        #     "reden": "Budget overschrijding",
        #     "context": "Taak vereist extra resources",
        #     "urgentie": "hoog"
        # }
        ```
    """

    name: str = "Escaleer"
    description: str = ""
    args_schema: type[BaseModel] = EscaleerSchema

    # Configuratie
    agent_id: UUID = Field(
        ...,
        description="ID van de agent die deze tool gebruikt",
    )
    organisatie: Any = Field(
        default=None,
        description="Referentie naar OrganisatieHierarchie",
    )
    escalatie_manager: Any = Field(
        default=None,
        description="Referentie naar EscalatieManager",
    )

    def __init__(
        self,
        agent_id: UUID,
        organisatie: "OrganisatieHierarchie | None" = None,
        escalatie_manager: "EscalatieManager | None" = None,
        **kwargs: Any,
    ) -> None:
        """Initialiseer de escalatie tool.

        Args:
            agent_id: ID van de agent die deze tool gebruikt.
            organisatie: Optionele organisatie voor het vinden van managers.
            escalatie_manager: Optionele escalatie manager.
            **kwargs: Extra argumenten voor BaseTool.
        """
        description = """Escaleer een probleem naar hoger management.

Gebruik deze tool wanneer je:
- Een probleem tegenkomt dat je niet zelf kunt oplossen
- Extra autorisatie of goedkeuring nodig hebt
- Geblokkeerd bent en hulp nodig hebt
- Een beslissing nodig is van hoger niveau

De input moet bevatten:
- reden: Waarom je escaleert
- context: Alle relevante informatie over het probleem
- urgentie: laag/normaal/hoog/kritiek (optioneel, standaard: normaal)

De escalatie wordt doorgestuurd naar je directe manager of hoger indien nodig."""

        super().__init__(
            agent_id=agent_id,
            organisatie=organisatie,
            escalatie_manager=escalatie_manager,
            description=description,
            **kwargs,
        )

    def _run(
        self,
        reden: str,
        context: str,
        urgentie: str = "normaal",
        **kwargs: Any,
    ) -> str:
        """Escaleer een probleem naar management.

        Args:
            reden: Reden voor escalatie.
            context: Context informatie.
            urgentie: Urgentie niveau.
            **kwargs: Extra argumenten (genegeerd).

        Returns:
            Bevestiging of foutmelding.
        """
        # Map urgentie naar prioriteit
        urgentie_mapping = {
            "laag": 0,
            "normaal": 1,
            "hoog": 5,
            "kritiek": 10,
        }
        prioriteit = urgentie_mapping.get(urgentie.lower(), 1)

        # Bepaal escalatiedoel
        doel_id = None
        doel_beschrijving = "management"

        if self.organisatie is not None:
            # Zoek directe manager
            manager = self.organisatie.krijg_directe_manager(self.agent_id)
            if manager:
                doel_id = manager
                doel_beschrijving = "directe manager"
            else:
                # Probeer management keten
                keten = self.organisatie.krijg_management_keten(self.agent_id)
                if keten:
                    doel_id = keten[0]
                    doel_beschrijving = "eerste beschikbare manager"

        # Maak escalatie via manager indien beschikbaar
        if self.escalatie_manager is not None and doel_id is not None:
            try:
                from crewai.governance.escalation import EscalatieTriggerType

                escalatie = self.escalatie_manager.escaleer(
                    bron_id=self.agent_id,
                    bron_type="agent",
                    regel=None,  # Handmatige escalatie
                    reden=reden,
                    context={
                        "beschrijving": context,
                        "urgentie": urgentie,
                        "handmatige_escalatie": True,
                    },
                )

                return f"""Escalatie succesvol aangemaakt.

Escalatie ID: {escalatie.id}
Geëscaleerd naar: {doel_beschrijving}
Reden: {reden}
Urgentie: {urgentie}
Status: {escalatie.status.value}

Je manager is op de hoogte gebracht en zal reageren op de escalatie."""

            except Exception as e:
                return f"Fout bij aanmaken escalatie: {str(e)}"

        # Zonder manager of doel, alleen bevestiging
        if doel_id is None:
            return f"""Escalatie geregistreerd, maar geen manager gevonden.

Reden: {reden}
Context: {context}
Urgentie: {urgentie}

Let op: Geen manager geconfigureerd in de organisatiehiërarchie.
Neem handmatig contact op met je leidinggevende."""

        return f"""Escalatie geregistreerd.

Geëscaleerd naar: {doel_beschrijving}
Reden: {reden}
Urgentie: {urgentie}

Let op: Geen escalatie manager geconfigureerd - escalatie wordt niet formeel gevolgd."""
