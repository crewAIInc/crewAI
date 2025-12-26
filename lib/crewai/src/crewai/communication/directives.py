"""Opdrachten en instructies systeem.

Dit module implementeert het opdrachtsysteem waarmee managers
taken kunnen delegeren naar ondergeschikten in de organisatie.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from crewai.organization.hierarchy import OrganisatieHierarchie


class OpdrachtStatus(str, Enum):
    """Status van een opdracht."""

    NIEUW = "nieuw"
    """Opdracht is aangemaakt maar nog niet geaccepteerd."""

    GEACCEPTEERD = "geaccepteerd"
    """Opdracht is geaccepteerd door ontvanger."""

    IN_UITVOERING = "in_uitvoering"
    """Opdracht wordt momenteel uitgevoerd."""

    WACHT_OP_GOEDKEURING = "wacht_op_goedkeuring"
    """Resultaat wacht op goedkeuring."""

    VOLTOOID = "voltooid"
    """Opdracht is succesvol voltooid."""

    GEWEIGERD = "geweigerd"
    """Opdracht is geweigerd door ontvanger."""

    GEANNULEERD = "geannuleerd"
    """Opdracht is geannuleerd door opdrachtgever."""

    GEESCALEERD = "geescaleerd"
    """Opdracht is geëscaleerd naar hoger niveau."""

    MISLUKT = "mislukt"
    """Opdracht is mislukt."""


class OpdrachtPrioriteit(str, Enum):
    """Prioriteitsniveau van een opdracht."""

    LAAG = "laag"
    """Lage prioriteit, kan wachten."""

    NORMAAL = "normaal"
    """Normale prioriteit."""

    HOOG = "hoog"
    """Hoge prioriteit, voorrang geven."""

    KRITIEK = "kritiek"
    """Kritieke prioriteit, direct uitvoeren."""


class OpdrachtVoortgang(BaseModel):
    """Voortgangsupdate voor een opdracht."""

    tijdstip: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van de update",
    )
    percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Voortgangspercentage (0-100)",
    )
    bericht: str = Field(
        ...,
        description="Voortgangsbericht",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra details",
    )


class Opdracht(BaseModel):
    """Een opdracht van een manager naar een medewerker.

    Representeert een taak of instructie die gedelegeerd wordt
    binnen de organisatiehiërarchie.

    Voorbeeld:
        ```python
        from crewai.communication import Opdracht, OpdrachtPrioriteit

        opdracht = Opdracht(
            afzender_id=manager_id,
            ontvanger_id=medewerker_id,
            titel="Analyseer Q4 resultaten",
            beschrijving="Maak een analyse van de Q4 resultaten",
            prioriteit=OpdrachtPrioriteit.HOOG,
            deadline=datetime(2024, 1, 15)
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor deze opdracht",
    )

    # Partijen
    afzender_id: UUID = Field(
        ...,
        description="ID van de opdrachtgever",
    )
    ontvanger_id: UUID = Field(
        ...,
        description="ID van de opdrachtnemer (agent of crew)",
    )
    ontvanger_type: Literal["agent", "crew"] = Field(
        default="agent",
        description="Type ontvanger",
    )

    # Inhoud
    titel: str = Field(
        ...,
        description="Titel van de opdracht",
        min_length=1,
    )
    beschrijving: str = Field(
        ...,
        description="Gedetailleerde beschrijving van de opdracht",
    )
    verwachte_output: str | None = Field(
        default=None,
        description="Beschrijving van de verwachte output",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context en informatie",
    )

    # Prioriteit en timing
    prioriteit: OpdrachtPrioriteit = Field(
        default=OpdrachtPrioriteit.NORMAAL,
        description="Prioriteit van de opdracht",
    )
    deadline: datetime | None = Field(
        default=None,
        description="Deadline voor de opdracht",
    )

    # Status
    status: OpdrachtStatus = Field(
        default=OpdrachtStatus.NIEUW,
        description="Huidige status van de opdracht",
    )

    # Tracking tijdstippen
    aangemaakt_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van aanmaken",
    )
    geaccepteerd_op: datetime | None = Field(
        default=None,
        description="Tijdstip van acceptatie",
    )
    gestart_op: datetime | None = Field(
        default=None,
        description="Tijdstip van start uitvoering",
    )
    voltooid_op: datetime | None = Field(
        default=None,
        description="Tijdstip van voltooiing",
    )

    # Voortgang
    voortgang_updates: list[OpdrachtVoortgang] = Field(
        default_factory=list,
        description="Lijst van voortgangsupdates",
    )
    huidig_voortgang_percentage: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Huidige voortgang in percentage",
    )

    # Resultaat
    resultaat: str | None = Field(
        default=None,
        description="Resultaat van de opdracht",
    )
    resultaat_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Gedetailleerd resultaat",
    )

    # Feedback
    feedback: str | None = Field(
        default=None,
        description="Feedback van de opdrachtgever",
    )

    # Gerelateerde opdrachten
    parent_opdracht_id: UUID | None = Field(
        default=None,
        description="ID van bovenliggende opdracht (voor sub-opdrachten)",
    )
    sub_opdracht_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs van sub-opdrachten",
    )

    # === STATUS METHODES ===

    def accepteer(self) -> None:
        """Accepteer de opdracht."""
        if self.status != OpdrachtStatus.NIEUW:
            raise ValueError(f"Kan opdracht niet accepteren in status {self.status}")
        self.status = OpdrachtStatus.GEACCEPTEERD
        self.geaccepteerd_op = datetime.now(timezone.utc)

    def weiger(self, reden: str | None = None) -> None:
        """Weiger de opdracht.

        Args:
            reden: Optionele reden voor weigering.
        """
        if self.status != OpdrachtStatus.NIEUW:
            raise ValueError(f"Kan opdracht niet weigeren in status {self.status}")
        self.status = OpdrachtStatus.GEWEIGERD
        if reden:
            self.feedback = reden

    def start(self) -> None:
        """Start de uitvoering van de opdracht."""
        if self.status not in [OpdrachtStatus.NIEUW, OpdrachtStatus.GEACCEPTEERD]:
            raise ValueError(f"Kan opdracht niet starten in status {self.status}")
        self.status = OpdrachtStatus.IN_UITVOERING
        self.gestart_op = datetime.now(timezone.utc)

    def update_voortgang(
        self,
        percentage: int,
        bericht: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Update de voortgang van de opdracht.

        Args:
            percentage: Voortgangspercentage (0-100).
            bericht: Voortgangsbericht.
            details: Optionele extra details.
        """
        if self.status != OpdrachtStatus.IN_UITVOERING:
            raise ValueError(
                f"Kan voortgang niet updaten in status {self.status}"
            )

        update = OpdrachtVoortgang(
            percentage=percentage,
            bericht=bericht,
            details=details or {},
        )
        self.voortgang_updates.append(update)
        self.huidig_voortgang_percentage = percentage

    def voltooi(self, resultaat: str, details: dict[str, Any] | None = None) -> None:
        """Voltooi de opdracht.

        Args:
            resultaat: Het resultaat van de opdracht.
            details: Optionele resultaat details.
        """
        if self.status != OpdrachtStatus.IN_UITVOERING:
            raise ValueError(f"Kan opdracht niet voltooien in status {self.status}")

        self.status = OpdrachtStatus.VOLTOOID
        self.voltooid_op = datetime.now(timezone.utc)
        self.resultaat = resultaat
        self.resultaat_details = details or {}
        self.huidig_voortgang_percentage = 100

    def markeer_mislukt(self, reden: str) -> None:
        """Markeer de opdracht als mislukt.

        Args:
            reden: Reden voor mislukking.
        """
        self.status = OpdrachtStatus.MISLUKT
        self.voltooid_op = datetime.now(timezone.utc)
        self.resultaat = reden

    def annuleer(self, reden: str | None = None) -> None:
        """Annuleer de opdracht.

        Args:
            reden: Optionele reden voor annulering.
        """
        if self.status in [OpdrachtStatus.VOLTOOID, OpdrachtStatus.GEANNULEERD]:
            raise ValueError(f"Kan opdracht niet annuleren in status {self.status}")
        self.status = OpdrachtStatus.GEANNULEERD
        if reden:
            self.feedback = reden

    def escaleer(self) -> None:
        """Markeer de opdracht als geëscaleerd."""
        self.status = OpdrachtStatus.GEESCALEERD

    # === QUERY METHODES ===

    def is_open(self) -> bool:
        """Check of de opdracht nog open is.

        Returns:
            True als de opdracht nog open is.
        """
        return self.status in [
            OpdrachtStatus.NIEUW,
            OpdrachtStatus.GEACCEPTEERD,
            OpdrachtStatus.IN_UITVOERING,
            OpdrachtStatus.WACHT_OP_GOEDKEURING,
        ]

    def is_voltooid(self) -> bool:
        """Check of de opdracht voltooid is.

        Returns:
            True als voltooid.
        """
        return self.status == OpdrachtStatus.VOLTOOID

    def is_te_laat(self) -> bool:
        """Check of de opdracht te laat is.

        Returns:
            True als de deadline is verstreken en niet voltooid.
        """
        if self.deadline is None:
            return False
        if self.status == OpdrachtStatus.VOLTOOID:
            return False
        return datetime.now(timezone.utc) > self.deadline

    def krijg_doorlooptijd(self) -> float | None:
        """Krijg de doorlooptijd in seconden.

        Returns:
            Doorlooptijd in seconden of None als niet voltooid.
        """
        if self.voltooid_op is None:
            return None
        return (self.voltooid_op - self.aangemaakt_op).total_seconds()


class OpdrachtManager(BaseModel):
    """Beheert opdrachten tussen crews en agents.

    Deze klasse coördineert het creëren, toewijzen en volgen
    van opdrachten in de organisatie.

    Voorbeeld:
        ```python
        from crewai.communication import OpdrachtManager, OpdrachtPrioriteit

        manager = OpdrachtManager()

        # Geef opdracht
        opdracht = manager.geef_opdracht(
            van_id=manager_agent_id,
            naar_id=medewerker_id,
            titel="Maak rapport",
            beschrijving="Maak een rapport van de maandelijkse cijfers",
            prioriteit=OpdrachtPrioriteit.HOOG
        )

        # Volg voortgang
        manager.update_voortgang(opdracht.id, 50, "Halverwege")

        # Voltooi
        manager.voltooi_opdracht(opdracht.id, "Rapport is klaar")
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    opdrachten: dict[UUID, Opdracht] = Field(
        default_factory=dict,
        description="Alle opdrachten geïndexeerd op ID",
    )

    # Organisatie referentie voor permissiecontroles
    organisatie: "OrganisatieHierarchie | None" = Field(
        default=None,
        description="Referentie naar organisatie voor permissies",
    )

    # Callbacks
    on_nieuwe_opdracht: Callable[[Opdracht], None] | None = Field(
        default=None,
        description="Callback bij nieuwe opdracht",
    )
    on_status_wijziging: Callable[[Opdracht, OpdrachtStatus], None] | None = Field(
        default=None,
        description="Callback bij status wijziging",
    )

    def geef_opdracht(
        self,
        van_id: UUID,
        naar_id: UUID,
        titel: str,
        beschrijving: str,
        prioriteit: OpdrachtPrioriteit = OpdrachtPrioriteit.NORMAAL,
        deadline: datetime | None = None,
        verwachte_output: str | None = None,
        context: dict[str, Any] | None = None,
        ontvanger_type: Literal["agent", "crew"] = "agent",
        parent_opdracht_id: UUID | None = None,
    ) -> Opdracht:
        """Maak en registreer een nieuwe opdracht.

        Args:
            van_id: ID van de opdrachtgever.
            naar_id: ID van de opdrachtnemer.
            titel: Titel van de opdracht.
            beschrijving: Beschrijving van de opdracht.
            prioriteit: Prioriteit van de opdracht.
            deadline: Optionele deadline.
            verwachte_output: Beschrijving van verwachte output.
            context: Extra context.
            ontvanger_type: Type ontvanger (agent of crew).
            parent_opdracht_id: Optioneel parent opdracht ID.

        Returns:
            De nieuwe opdracht.
        """
        # Controleer permissies indien organisatie beschikbaar
        if self.organisatie is not None:
            if not self.organisatie.mag_opdracht_geven(van_id, naar_id):
                raise PermissionError(
                    f"Agent {van_id} mag geen opdrachten geven aan {naar_id}"
                )

        opdracht = Opdracht(
            afzender_id=van_id,
            ontvanger_id=naar_id,
            ontvanger_type=ontvanger_type,
            titel=titel,
            beschrijving=beschrijving,
            prioriteit=prioriteit,
            deadline=deadline,
            verwachte_output=verwachte_output,
            context=context or {},
            parent_opdracht_id=parent_opdracht_id,
        )

        self.opdrachten[opdracht.id] = opdracht

        # Update parent indien aanwezig
        if parent_opdracht_id and parent_opdracht_id in self.opdrachten:
            self.opdrachten[parent_opdracht_id].sub_opdracht_ids.append(opdracht.id)

        # Trigger callback
        if self.on_nieuwe_opdracht:
            self.on_nieuwe_opdracht(opdracht)

        return opdracht

    def krijg_opdracht(self, opdracht_id: UUID) -> Opdracht | None:
        """Krijg een opdracht op ID.

        Args:
            opdracht_id: ID van de opdracht.

        Returns:
            De opdracht of None.
        """
        return self.opdrachten.get(opdracht_id)

    def accepteer_opdracht(self, opdracht_id: UUID) -> Opdracht | None:
        """Accepteer een opdracht.

        Args:
            opdracht_id: ID van de opdracht.

        Returns:
            De bijgewerkte opdracht of None.
        """
        opdracht = self.opdrachten.get(opdracht_id)
        if opdracht is None:
            return None

        oude_status = opdracht.status
        opdracht.accepteer()
        self._trigger_status_wijziging(opdracht, oude_status)
        return opdracht

    def weiger_opdracht(
        self, opdracht_id: UUID, reden: str | None = None
    ) -> Opdracht | None:
        """Weiger een opdracht.

        Args:
            opdracht_id: ID van de opdracht.
            reden: Optionele reden.

        Returns:
            De bijgewerkte opdracht of None.
        """
        opdracht = self.opdrachten.get(opdracht_id)
        if opdracht is None:
            return None

        oude_status = opdracht.status
        opdracht.weiger(reden)
        self._trigger_status_wijziging(opdracht, oude_status)
        return opdracht

    def start_opdracht(self, opdracht_id: UUID) -> Opdracht | None:
        """Start een opdracht.

        Args:
            opdracht_id: ID van de opdracht.

        Returns:
            De bijgewerkte opdracht of None.
        """
        opdracht = self.opdrachten.get(opdracht_id)
        if opdracht is None:
            return None

        oude_status = opdracht.status
        opdracht.start()
        self._trigger_status_wijziging(opdracht, oude_status)
        return opdracht

    def update_voortgang(
        self,
        opdracht_id: UUID,
        percentage: int,
        bericht: str,
        details: dict[str, Any] | None = None,
    ) -> Opdracht | None:
        """Update de voortgang van een opdracht.

        Args:
            opdracht_id: ID van de opdracht.
            percentage: Voortgangspercentage.
            bericht: Voortgangsbericht.
            details: Optionele details.

        Returns:
            De bijgewerkte opdracht of None.
        """
        opdracht = self.opdrachten.get(opdracht_id)
        if opdracht is None:
            return None

        opdracht.update_voortgang(percentage, bericht, details)
        return opdracht

    def voltooi_opdracht(
        self,
        opdracht_id: UUID,
        resultaat: str,
        details: dict[str, Any] | None = None,
    ) -> Opdracht | None:
        """Voltooi een opdracht.

        Args:
            opdracht_id: ID van de opdracht.
            resultaat: Het resultaat.
            details: Optionele details.

        Returns:
            De bijgewerkte opdracht of None.
        """
        opdracht = self.opdrachten.get(opdracht_id)
        if opdracht is None:
            return None

        oude_status = opdracht.status
        opdracht.voltooi(resultaat, details)
        self._trigger_status_wijziging(opdracht, oude_status)
        return opdracht

    def annuleer_opdracht(
        self, opdracht_id: UUID, reden: str | None = None
    ) -> Opdracht | None:
        """Annuleer een opdracht.

        Args:
            opdracht_id: ID van de opdracht.
            reden: Optionele reden.

        Returns:
            De bijgewerkte opdracht of None.
        """
        opdracht = self.opdrachten.get(opdracht_id)
        if opdracht is None:
            return None

        oude_status = opdracht.status
        opdracht.annuleer(reden)
        self._trigger_status_wijziging(opdracht, oude_status)
        return opdracht

    def _trigger_status_wijziging(
        self, opdracht: Opdracht, oude_status: OpdrachtStatus
    ) -> None:
        """Trigger de status wijziging callback.

        Args:
            opdracht: De opdracht.
            oude_status: De oude status.
        """
        if self.on_status_wijziging and opdracht.status != oude_status:
            self.on_status_wijziging(opdracht, oude_status)

    # === QUERY METHODES ===

    def krijg_opdrachten_voor(
        self,
        ontvanger_id: UUID,
        alleen_open: bool = True,
    ) -> list[Opdracht]:
        """Krijg alle opdrachten voor een ontvanger.

        Args:
            ontvanger_id: ID van de ontvanger.
            alleen_open: Als True, alleen open opdrachten.

        Returns:
            Lijst van opdrachten.
        """
        opdrachten = [
            o for o in self.opdrachten.values() if o.ontvanger_id == ontvanger_id
        ]

        if alleen_open:
            opdrachten = [o for o in opdrachten if o.is_open()]

        return sorted(
            opdrachten,
            key=lambda o: (o.prioriteit.value, o.aangemaakt_op),
            reverse=True,
        )

    def krijg_opdrachten_van(
        self,
        afzender_id: UUID,
        alleen_open: bool = True,
    ) -> list[Opdracht]:
        """Krijg alle opdrachten van een afzender.

        Args:
            afzender_id: ID van de afzender.
            alleen_open: Als True, alleen open opdrachten.

        Returns:
            Lijst van opdrachten.
        """
        opdrachten = [
            o for o in self.opdrachten.values() if o.afzender_id == afzender_id
        ]

        if alleen_open:
            opdrachten = [o for o in opdrachten if o.is_open()]

        return sorted(opdrachten, key=lambda o: o.aangemaakt_op, reverse=True)

    def krijg_te_late_opdrachten(self) -> list[Opdracht]:
        """Krijg alle opdrachten die te laat zijn.

        Returns:
            Lijst van te late opdrachten.
        """
        return [o for o in self.opdrachten.values() if o.is_te_laat()]

    def krijg_opdrachten_per_status(
        self, status: OpdrachtStatus
    ) -> list[Opdracht]:
        """Krijg alle opdrachten met een bepaalde status.

        Args:
            status: De status om op te filteren.

        Returns:
            Lijst van opdrachten.
        """
        return [o for o in self.opdrachten.values() if o.status == status]

    def krijg_statistieken(self) -> dict[str, Any]:
        """Krijg statistieken over opdrachten.

        Returns:
            Dictionary met statistieken.
        """
        alle = list(self.opdrachten.values())
        voltooide = [o for o in alle if o.is_voltooid()]

        doorlooptijden = [o.krijg_doorlooptijd() for o in voltooide]
        doorlooptijden = [d for d in doorlooptijden if d is not None]

        return {
            "totaal": len(alle),
            "open": len([o for o in alle if o.is_open()]),
            "voltooid": len(voltooide),
            "te_laat": len([o for o in alle if o.is_te_laat()]),
            "gemiddelde_doorlooptijd_seconden": (
                sum(doorlooptijden) / len(doorlooptijden) if doorlooptijden else 0
            ),
            "per_status": {
                status.value: len([o for o in alle if o.status == status])
                for status in OpdrachtStatus
            },
            "per_prioriteit": {
                prioriteit.value: len(
                    [o for o in alle if o.prioriteit == prioriteit]
                )
                for prioriteit in OpdrachtPrioriteit
            },
        }
