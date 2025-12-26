"""Manager klasse voor organisatie-gerelateerde tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.tools.organization_tools.escalate_tool import EscaleerTool
from crewai.tools.organization_tools.give_directive_tool import GeefOpdrachtTool
from crewai.tools.organization_tools.report_tool import RapporteerTool

if TYPE_CHECKING:
    from crewai.communication.directives import OpdrachtManager
    from crewai.communication.reports import RapportManager
    from crewai.governance.escalation import EscalatieManager
    from crewai.organization.hierarchy import OrganisatieHierarchie
    from crewai.organization.role import Rol


class OrganizationTools(BaseModel):
    """Manager voor organisatie tools.

    Deze klasse creëert en beheert tools waarmee agents kunnen
    communiceren binnen de organisatiehiërarchie, inclusief
    het geven van opdrachten, rapporteren, en escaleren.

    Voorbeeld:
        ```python
        from crewai.tools.organization_tools import OrganizationTools

        # Maak tools voor een agent
        org_tools = OrganizationTools(
            agent_id=agent_id,
            organisatie=org,
            opdracht_manager=opdracht_mgr,
            rapport_manager=rapport_mgr,
            escalatie_manager=escalatie_mgr,
            ondergeschikten={"team_a": team_a_id},
            managers={"directeur": directeur_id}
        )

        # Krijg alle beschikbare tools
        tools = org_tools.tools()

        # Voeg toe aan agent
        agent.tools.extend(tools)
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    agent_id: UUID = Field(
        ...,
        description="ID van de agent waarvoor tools worden aangemaakt",
    )
    rol: Any = Field(
        default=None,
        description="Rol van de agent (bepaalt welke tools beschikbaar zijn)",
    )
    organisatie: Any = Field(
        default=None,
        description="Referentie naar OrganisatieHierarchie",
    )
    opdracht_manager: Any = Field(
        default=None,
        description="Referentie naar OpdrachtManager",
    )
    rapport_manager: Any = Field(
        default=None,
        description="Referentie naar RapportManager",
    )
    escalatie_manager: Any = Field(
        default=None,
        description="Referentie naar EscalatieManager",
    )
    ondergeschikten: dict[str, UUID] = Field(
        default_factory=dict,
        description="Mapping van namen naar IDs van ondergeschikten",
    )
    managers: dict[str, UUID] = Field(
        default_factory=dict,
        description="Mapping van namen naar IDs van managers",
    )

    def tools(self) -> list[BaseTool]:
        """Geef alle beschikbare organisatie tools voor deze agent.

        Welke tools beschikbaar zijn hangt af van de rol en permissies
        van de agent.

        Returns:
            Lijst met BaseTool instances.
        """
        tools: list[BaseTool] = []

        # Check of agent opdrachten mag geven
        kan_opdrachten_geven = self._mag_opdrachten_geven()
        if kan_opdrachten_geven and self.ondergeschikten:
            tools.append(
                GeefOpdrachtTool(
                    agent_id=self.agent_id,
                    organisatie=self.organisatie,
                    opdracht_manager=self.opdracht_manager,
                    ontvangers=self.ondergeschikten,
                )
            )

        # Rapporteer tool is altijd beschikbaar (naar managers)
        if self.managers or self.organisatie:
            tools.append(
                RapporteerTool(
                    agent_id=self.agent_id,
                    organisatie=self.organisatie,
                    rapport_manager=self.rapport_manager,
                    ontvangers=self.managers,
                )
            )

        # Check of agent mag escaleren
        kan_escaleren = self._mag_escaleren()
        if kan_escaleren:
            tools.append(
                EscaleerTool(
                    agent_id=self.agent_id,
                    organisatie=self.organisatie,
                    escalatie_manager=self.escalatie_manager,
                )
            )

        return tools

    def _mag_opdrachten_geven(self) -> bool:
        """Check of de agent opdrachten mag geven.

        Returns:
            True als opdrachten geven is toegestaan.
        """
        # Check rol indien beschikbaar
        if self.rol is not None:
            return getattr(self.rol, "kan_opdrachten_geven", False)

        # Check organisatie indien beschikbaar
        if self.organisatie is not None:
            rol = self.organisatie.krijg_rol(self.agent_id)
            if rol is not None:
                return rol.kan_opdrachten_geven

        # Standaard: ja als er ondergeschikten zijn
        return bool(self.ondergeschikten)

    def _mag_escaleren(self) -> bool:
        """Check of de agent mag escaleren.

        Returns:
            True als escaleren is toegestaan.
        """
        # Check rol indien beschikbaar
        if self.rol is not None:
            return getattr(self.rol, "kan_escaleren", True)

        # Check organisatie indien beschikbaar
        if self.organisatie is not None:
            rol = self.organisatie.krijg_rol(self.agent_id)
            if rol is not None:
                return rol.kan_escaleren

        # Standaard: ja (iedereen mag escaleren)
        return True

    @classmethod
    def voor_manager(
        cls,
        agent_id: UUID,
        organisatie: "OrganisatieHierarchie",
        ondergeschikten: dict[str, UUID],
        opdracht_manager: "OpdrachtManager | None" = None,
        rapport_manager: "RapportManager | None" = None,
        escalatie_manager: "EscalatieManager | None" = None,
    ) -> "OrganizationTools":
        """Maak tools voor een manager agent.

        Args:
            agent_id: ID van de manager agent.
            organisatie: De organisatiehiërarchie.
            ondergeschikten: Mapping van namen naar ondergeschikte IDs.
            opdracht_manager: Optionele opdracht manager.
            rapport_manager: Optionele rapport manager.
            escalatie_manager: Optionele escalatie manager.

        Returns:
            OrganizationTools instance voor een manager.
        """
        # Zoek managers van deze manager
        managers: dict[str, UUID] = {}
        keten = organisatie.krijg_management_keten(agent_id)
        for i, manager_id in enumerate(keten[:3]):  # Max 3 niveaus
            managers[f"manager_{i+1}"] = manager_id

        return cls(
            agent_id=agent_id,
            organisatie=organisatie,
            opdracht_manager=opdracht_manager,
            rapport_manager=rapport_manager,
            escalatie_manager=escalatie_manager,
            ondergeschikten=ondergeschikten,
            managers=managers,
        )

    @classmethod
    def voor_medewerker(
        cls,
        agent_id: UUID,
        organisatie: "OrganisatieHierarchie",
        rapport_manager: "RapportManager | None" = None,
        escalatie_manager: "EscalatieManager | None" = None,
    ) -> "OrganizationTools":
        """Maak tools voor een medewerker agent (geen ondergeschikten).

        Args:
            agent_id: ID van de medewerker agent.
            organisatie: De organisatiehiërarchie.
            rapport_manager: Optionele rapport manager.
            escalatie_manager: Optionele escalatie manager.

        Returns:
            OrganizationTools instance voor een medewerker.
        """
        # Zoek managers
        managers: dict[str, UUID] = {}
        direct_manager = organisatie.krijg_directe_manager(agent_id)
        if direct_manager:
            managers["manager"] = direct_manager

        return cls(
            agent_id=agent_id,
            organisatie=organisatie,
            rapport_manager=rapport_manager,
            escalatie_manager=escalatie_manager,
            ondergeschikten={},  # Medewerker heeft geen ondergeschikten
            managers=managers,
        )
