"""Afdelingen en departementen in de organisatie.

Dit module definieert de Afdeling klasse waarmee organisaties
kunnen worden opgedeeld in logische eenheden met eigen
resources, leden en configuratie.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from crewai.organization.role import Rol


IsolatieNiveau = Literal["open", "afdeling", "strikt"]
"""Type voor isolatieniveaus tussen afdelingen.

- open: Iedereen kan communiceren met deze afdeling
- afdeling: Alleen binnen afdeling en met management
- strikt: Alleen expliciete toegang toegestaan
"""


class AfdelingsResource(BaseModel):
    """Resource limieten voor een afdeling.

    Definieert de beschikbare resources en limieten
    voor een specifieke afdeling.
    """

    max_rpm: int | None = Field(
        default=None,
        description="Maximum API requests per minuut (None = onbeperkt)",
    )
    max_tokens_per_dag: int | None = Field(
        default=None,
        description="Maximum tokens per dag (None = onbeperkt)",
    )
    budget_limiet: float | None = Field(
        default=None,
        description="Maximaal budget in EUR (None = onbeperkt)",
    )
    budget_gebruikt: float = Field(
        default=0.0,
        description="Huidig verbruikt budget",
    )
    max_gelijktijdige_crews: int = Field(
        default=10,
        description="Maximum aantal crews dat tegelijk actief mag zijn",
    )

    def is_binnen_budget(self, bedrag: float) -> bool:
        """Controleer of een bedrag binnen het budget past.

        Args:
            bedrag: Het bedrag om te controleren.

        Returns:
            True als het bedrag binnen budget past.
        """
        if self.budget_limiet is None:
            return True
        return (self.budget_gebruikt + bedrag) <= self.budget_limiet

    def registreer_uitgave(self, bedrag: float) -> bool:
        """Registreer een uitgave tegen het budget.

        Args:
            bedrag: Het uit te geven bedrag.

        Returns:
            True als de uitgave geregistreerd is, False als budget overschreden.
        """
        if not self.is_binnen_budget(bedrag):
            return False
        self.budget_gebruikt += bedrag
        return True


class Afdeling(BaseModel):
    """Representeert een afdeling in de organisatie.

    Een afdeling is een logische groepering van agents en crews
    met eigen resources, configuratie en isolatieregels.

    Voorbeeld:
        ```python
        from crewai.organization import Afdeling

        trading = Afdeling(
            naam="Trading",
            beschrijving="Alle handelsactiviteiten",
            isolatie_niveau="afdeling",
            resources=AfdelingsResource(
                budget_limiet=10000.0,
                max_rpm=100
            )
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor deze afdeling",
        frozen=True,
    )
    naam: str = Field(
        ...,
        description="Naam van de afdeling",
        min_length=1,
    )
    beschrijving: str | None = Field(
        default=None,
        description="Beschrijving van de afdeling en verantwoordelijkheden",
    )

    # Hiërarchie
    parent_afdeling_id: UUID | None = Field(
        default=None,
        description="ID van de bovenliggende afdeling (voor sub-afdelingen)",
    )
    manager_agent_id: UUID | None = Field(
        default=None,
        description="ID van de agent die deze afdeling leidt",
    )

    # Leden
    agent_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs van agents die in deze afdeling werken",
    )
    crew_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs van crews die bij deze afdeling horen",
    )

    # Configuratie
    isolatie_niveau: IsolatieNiveau = Field(
        default="afdeling",
        description="Hoe geïsoleerd is deze afdeling van anderen",
    )
    toegestane_afdelingen: list[UUID] = Field(
        default_factory=list,
        description="IDs van afdelingen waarmee gecommuniceerd mag worden (bij strikt niveau)",
    )

    # Resources
    resources: AfdelingsResource = Field(
        default_factory=AfdelingsResource,
        description="Resource limieten voor deze afdeling",
    )

    # Status
    actief: bool = Field(
        default=True,
        description="Of deze afdeling momenteel actief is",
    )

    # Tracking
    aangemaakt_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van aanmaken",
    )
    laatst_gewijzigd_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van laatste wijziging",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata voor de afdeling",
    )

    @field_validator("agent_ids", "crew_ids", "toegestane_afdelingen")
    @classmethod
    def valideer_unieke_ids(cls, v: list[UUID]) -> list[UUID]:
        """Zorg dat alle IDs in de lijst uniek zijn."""
        return list(dict.fromkeys(v))

    def voeg_agent_toe(self, agent_id: UUID) -> None:
        """Voeg een agent toe aan deze afdeling.

        Args:
            agent_id: ID van de agent om toe te voegen.
        """
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
            self._update_gewijzigd()

    def verwijder_agent(self, agent_id: UUID) -> bool:
        """Verwijder een agent uit deze afdeling.

        Args:
            agent_id: ID van de agent om te verwijderen.

        Returns:
            True als de agent verwijderd is, False als niet gevonden.
        """
        if agent_id in self.agent_ids:
            self.agent_ids.remove(agent_id)
            self._update_gewijzigd()
            return True
        return False

    def voeg_crew_toe(self, crew_id: UUID) -> None:
        """Voeg een crew toe aan deze afdeling.

        Args:
            crew_id: ID van de crew om toe te voegen.
        """
        if crew_id not in self.crew_ids:
            self.crew_ids.append(crew_id)
            self._update_gewijzigd()

    def verwijder_crew(self, crew_id: UUID) -> bool:
        """Verwijder een crew uit deze afdeling.

        Args:
            crew_id: ID van de crew om te verwijderen.

        Returns:
            True als de crew verwijderd is, False als niet gevonden.
        """
        if crew_id in self.crew_ids:
            self.crew_ids.remove(crew_id)
            self._update_gewijzigd()
            return True
        return False

    def mag_communiceren_met(self, andere_afdeling: "Afdeling") -> bool:
        """Controleer of deze afdeling mag communiceren met een andere afdeling.

        Args:
            andere_afdeling: De afdeling om te controleren.

        Returns:
            True als communicatie toegestaan is.
        """
        # Zelfde afdeling mag altijd
        if self.id == andere_afdeling.id:
            return True

        # Check isolatieniveau
        if self.isolatie_niveau == "open":
            return True

        if self.isolatie_niveau == "afdeling":
            # Alleen binnen dezelfde parent of direct gekoppeld
            return (
                self.parent_afdeling_id == andere_afdeling.id
                or andere_afdeling.parent_afdeling_id == self.id
                or self.parent_afdeling_id == andere_afdeling.parent_afdeling_id
            )

        if self.isolatie_niveau == "strikt":
            # Alleen expliciete toegang
            return andere_afdeling.id in self.toegestane_afdelingen

        return False

    def bevat_agent(self, agent_id: UUID) -> bool:
        """Controleer of een agent in deze afdeling werkt.

        Args:
            agent_id: ID van de agent om te controleren.

        Returns:
            True als de agent in deze afdeling werkt.
        """
        return agent_id in self.agent_ids

    def bevat_crew(self, crew_id: UUID) -> bool:
        """Controleer of een crew bij deze afdeling hoort.

        Args:
            crew_id: ID van de crew om te controleren.

        Returns:
            True als de crew bij deze afdeling hoort.
        """
        return crew_id in self.crew_ids

    def _update_gewijzigd(self) -> None:
        """Update het laatst_gewijzigd_op tijdstip."""
        object.__setattr__(self, "laatst_gewijzigd_op", datetime.now(timezone.utc))


class AfdelingsManager(BaseModel):
    """Beheert alle afdelingen in een organisatie.

    Biedt methodes voor het aanmaken, opvragen en beheren
    van afdelingen en hun onderlinge relaties.
    """

    model_config = {"arbitrary_types_allowed": True}

    afdelingen: dict[UUID, Afdeling] = Field(
        default_factory=dict,
        description="Alle afdelingen geïndexeerd op ID",
    )

    def voeg_toe(self, afdeling: Afdeling) -> None:
        """Voeg een afdeling toe.

        Args:
            afdeling: De afdeling om toe te voegen.
        """
        self.afdelingen[afdeling.id] = afdeling

    def verwijder(self, afdeling_id: UUID) -> bool:
        """Verwijder een afdeling.

        Args:
            afdeling_id: ID van de afdeling om te verwijderen.

        Returns:
            True als verwijderd, False als niet gevonden.
        """
        if afdeling_id in self.afdelingen:
            del self.afdelingen[afdeling_id]
            return True
        return False

    def krijg(self, afdeling_id: UUID) -> Afdeling | None:
        """Krijg een afdeling op ID.

        Args:
            afdeling_id: ID van de afdeling.

        Returns:
            De afdeling of None als niet gevonden.
        """
        return self.afdelingen.get(afdeling_id)

    def krijg_op_naam(self, naam: str) -> Afdeling | None:
        """Krijg een afdeling op naam.

        Args:
            naam: Naam van de afdeling.

        Returns:
            De afdeling of None als niet gevonden.
        """
        for afdeling in self.afdelingen.values():
            if afdeling.naam == naam:
                return afdeling
        return None

    def krijg_sub_afdelingen(self, parent_id: UUID) -> list[Afdeling]:
        """Krijg alle sub-afdelingen van een afdeling.

        Args:
            parent_id: ID van de parent afdeling.

        Returns:
            Lijst van sub-afdelingen.
        """
        return [
            afd
            for afd in self.afdelingen.values()
            if afd.parent_afdeling_id == parent_id
        ]

    def krijg_root_afdelingen(self) -> list[Afdeling]:
        """Krijg alle top-level afdelingen (zonder parent).

        Returns:
            Lijst van root afdelingen.
        """
        return [
            afd for afd in self.afdelingen.values() if afd.parent_afdeling_id is None
        ]

    def krijg_afdeling_voor_agent(self, agent_id: UUID) -> Afdeling | None:
        """Vind de afdeling waar een agent werkt.

        Args:
            agent_id: ID van de agent.

        Returns:
            De afdeling of None als niet gevonden.
        """
        for afdeling in self.afdelingen.values():
            if afdeling.bevat_agent(agent_id):
                return afdeling
        return None

    def krijg_afdeling_voor_crew(self, crew_id: UUID) -> Afdeling | None:
        """Vind de afdeling waar een crew bij hoort.

        Args:
            crew_id: ID van de crew.

        Returns:
            De afdeling of None als niet gevonden.
        """
        for afdeling in self.afdelingen.values():
            if afdeling.bevat_crew(crew_id):
                return afdeling
        return None

    def alle_namen(self) -> list[str]:
        """Krijg een lijst van alle afdelingsnamen.

        Returns:
            Lijst van namen.
        """
        return [afd.naam for afd in self.afdelingen.values()]
