"""Tool voor het geven van opdrachten aan ondergeschikten."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from crewai.communication.directives import OpdrachtManager, OpdrachtPrioriteit
    from crewai.organization.hierarchy import OrganisatieHierarchie


class GeefOpdrachtSchema(BaseModel):
    """Schema voor opdracht parameters."""

    ontvanger: str = Field(
        ...,
        description="Naam of ID van de ontvanger (crew of agent)",
    )
    opdracht: str = Field(
        ...,
        description="De opdracht of taak die uitgevoerd moet worden",
    )
    context: str = Field(
        ...,
        description="Alle benodigde context en achtergrondinformatie",
    )
    prioriteit: str = Field(
        default="normaal",
        description="Prioriteit: laag, normaal, hoog, of kritiek",
    )


class GeefOpdrachtTool(BaseTool):
    """Tool voor het geven van opdrachten aan ondergeschikte crews of agents.

    Hiermee kan een manager opdrachten delegeren naar medewerkers
    in de organisatiehiërarchie.

    Voorbeeld:
        ```python
        from crewai.tools.organization_tools import GeefOpdrachtTool

        tool = GeefOpdrachtTool(
            agent_id=manager_id,
            organisatie=org,
            opdracht_manager=opdracht_mgr,
            ontvangers={"trading": trading_crew_id}
        )

        # Agent kan nu opdrachten geven:
        # Actie: Geef Opdracht
        # Actie Input: {
        #     "ontvanger": "trading",
        #     "opdracht": "Koop 0.01 BTC",
        #     "context": "Marktprijs is gunstig"
        # }
        ```
    """

    name: str = "Geef Opdracht"
    description: str = ""
    args_schema: type[BaseModel] = GeefOpdrachtSchema

    # Configuratie
    agent_id: UUID = Field(
        ...,
        description="ID van de agent die deze tool gebruikt",
    )
    organisatie: Any = Field(
        default=None,
        description="Referentie naar OrganisatieHierarchie",
    )
    opdracht_manager: Any = Field(
        default=None,
        description="Referentie naar OpdrachtManager",
    )
    ontvangers: dict[str, UUID] = Field(
        default_factory=dict,
        description="Mapping van namen naar IDs van mogelijke ontvangers",
    )

    def __init__(
        self,
        agent_id: UUID,
        organisatie: "OrganisatieHierarchie | None" = None,
        opdracht_manager: "OpdrachtManager | None" = None,
        ontvangers: dict[str, UUID] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialiseer de opdracht tool.

        Args:
            agent_id: ID van de agent die deze tool gebruikt.
            organisatie: Optionele organisatie voor permissiecontrole.
            opdracht_manager: Optionele opdracht manager.
            ontvangers: Mapping van namen naar IDs.
            **kwargs: Extra argumenten voor BaseTool.
        """
        ontvangers = ontvangers or {}
        ontvanger_namen = ", ".join(ontvangers.keys()) if ontvangers else "niemand"

        description = f"""Geef een opdracht aan een ondergeschikte crew of agent.

Beschikbare ontvangers: {ontvanger_namen}

De input moet bevatten:
- ontvanger: Naam van de ontvanger (één van: {ontvanger_namen})
- opdracht: Wat moet er gedaan worden
- context: Alle benodigde achtergrondinformatie
- prioriteit: laag, normaal, hoog, of kritiek (optioneel, standaard: normaal)

Gebruik deze tool om werk te delegeren naar je teamleden."""

        super().__init__(
            agent_id=agent_id,
            organisatie=organisatie,
            opdracht_manager=opdracht_manager,
            ontvangers=ontvangers,
            description=description,
            **kwargs,
        )

    def _run(
        self,
        ontvanger: str,
        opdracht: str,
        context: str,
        prioriteit: str = "normaal",
        **kwargs: Any,
    ) -> str:
        """Geef een opdracht aan de gespecificeerde ontvanger.

        Args:
            ontvanger: Naam of ID van de ontvanger.
            opdracht: De opdracht beschrijving.
            context: Context informatie.
            prioriteit: Prioriteit niveau.
            **kwargs: Extra argumenten (genegeerd).

        Returns:
            Bevestiging of foutmelding.
        """
        # Zoek ontvanger ID
        ontvanger_id = self.ontvangers.get(ontvanger)
        if ontvanger_id is None:
            # Probeer als UUID
            try:
                ontvanger_id = UUID(ontvanger)
            except ValueError:
                beschikbaar = ", ".join(self.ontvangers.keys())
                return f"Fout: Ontvanger '{ontvanger}' niet gevonden. Beschikbare ontvangers: {beschikbaar}"

        # Controleer permissies
        if self.organisatie is not None:
            if not self.organisatie.mag_opdracht_geven(self.agent_id, ontvanger_id):
                return f"Fout: Je hebt geen toestemming om opdrachten te geven aan '{ontvanger}'"

        # Map prioriteit string naar enum
        prioriteit_mapping = {
            "laag": "laag",
            "normaal": "normaal",
            "hoog": "hoog",
            "kritiek": "kritiek",
        }
        prioriteit_waarde = prioriteit_mapping.get(prioriteit.lower(), "normaal")

        # Maak opdracht via manager indien beschikbaar
        if self.opdracht_manager is not None:
            try:
                from crewai.communication.directives import OpdrachtPrioriteit

                prioriteit_enum = OpdrachtPrioriteit(prioriteit_waarde)
                nieuwe_opdracht = self.opdracht_manager.geef_opdracht(
                    van_id=self.agent_id,
                    naar_id=ontvanger_id,
                    titel=opdracht[:100],  # Eerste 100 chars als titel
                    beschrijving=opdracht,
                    prioriteit=prioriteit_enum,
                    context={"achtergrond": context},
                )
                return f"""Opdracht succesvol verstuurd naar '{ontvanger}'.

Opdracht ID: {nieuwe_opdracht.id}
Titel: {nieuwe_opdracht.titel}
Prioriteit: {prioriteit_waarde}
Status: {nieuwe_opdracht.status.value}

De ontvanger zal de opdracht uitvoeren en rapporteren."""

            except PermissionError as e:
                return f"Fout: {str(e)}"
            except Exception as e:
                return f"Fout bij aanmaken opdracht: {str(e)}"

        # Zonder manager, alleen bevestiging
        return f"""Opdracht geregistreerd voor '{ontvanger}'.

Opdracht: {opdracht}
Context: {context}
Prioriteit: {prioriteit_waarde}

Let op: Geen opdracht manager geconfigureerd - opdracht wordt niet formeel gevolgd."""
