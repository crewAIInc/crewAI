"""Escalatiepaden en automatische escalatie.

Dit module implementeert escalatiemechanismen waarmee problemen,
timeouts, en budget-overschrijdingen automatisch kunnen worden
geëscaleerd naar het juiste management niveau.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from crewai.organization.hierarchy import OrganisatieHierarchie


class EscalatieTriggerType(str, Enum):
    """Type triggers voor automatische escalatie."""

    TIMEOUT = "timeout"
    """Escaleer na een tijdslimiet."""

    FOUT = "fout"
    """Escaleer bij een fout."""

    BUDGET_OVERSCHREDEN = "budget_overschreden"
    """Escaleer bij budget overschrijding."""

    HANDMATIG = "handmatig"
    """Handmatige escalatie door agent."""

    HERHAALDE_POGINGEN = "herhaalde_pogingen"
    """Escaleer na meerdere mislukte pogingen."""

    GEEN_VOORTGANG = "geen_voortgang"
    """Escaleer als er geen voortgang is."""


class EscalatieDoelType(str, Enum):
    """Naar wie wordt geëscaleerd."""

    DIRECTE_MANAGER = "directe_manager"
    """Escaleer naar directe manager."""

    AFDELING_HOOFD = "afdeling_hoofd"
    """Escaleer naar afdelingshoofd."""

    DIRECTIE = "directie"
    """Escaleer naar directie."""

    SPECIFIEK = "specifiek"
    """Escaleer naar specifieke entiteit."""

    VOLGENDE_IN_KETEN = "volgende_in_keten"
    """Escaleer naar volgende in management keten."""


class EscalatieActieType(str, Enum):
    """Type actie bij escalatie."""

    MELDEN = "melden"
    """Alleen melden, geen actie."""

    HERTOEWIJZEN = "hertoewijzen"
    """Taak hertoewijzen aan escalatiedoel."""

    STOPPEN = "stoppen"
    """Taak stoppen en escaleren."""

    GOEDKEURING_VRAGEN = "goedkeuring_vragen"
    """Vraag goedkeuring voor verder gaan."""

    PARALLEL_UITVOEREN = "parallel_uitvoeren"
    """Start parallelle uitvoering bij escalatiedoel."""


class EscalatieStatus(str, Enum):
    """Status van een escalatie."""

    NIEUW = "nieuw"
    """Escalatie is aangemaakt."""

    IN_BEHANDELING = "in_behandeling"
    """Escalatie wordt behandeld."""

    OPGELOST = "opgelost"
    """Escalatie is opgelost."""

    DOORGESTUURD = "doorgestuurd"
    """Escalatie is verder geëscaleerd."""

    GESLOTEN = "gesloten"
    """Escalatie is gesloten."""


class EscalatieRegel(BaseModel):
    """Regel die definieert wanneer en hoe te escaleren.

    Voorbeeld:
        ```python
        from crewai.governance import (
            EscalatieRegel,
            EscalatieTriggerType,
            EscalatieDoelType,
            EscalatieActieType
        )

        # Escaleer naar manager na 1 uur inactiviteit
        regel = EscalatieRegel(
            naam="Timeout escalatie",
            trigger_type=EscalatieTriggerType.TIMEOUT,
            trigger_waarde=3600,  # seconden
            escaleer_naar=EscalatieDoelType.DIRECTE_MANAGER,
            actie=EscalatieActieType.MELDEN
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor deze regel",
    )
    naam: str = Field(
        ...,
        description="Naam van de escalatieregel",
    )
    beschrijving: str | None = Field(
        default=None,
        description="Beschrijving van wanneer en waarom deze regel triggert",
    )

    # Trigger configuratie
    trigger_type: EscalatieTriggerType = Field(
        ...,
        description="Type trigger voor deze escalatie",
    )
    trigger_waarde: Any = Field(
        default=None,
        description="Waarde voor de trigger (bijv. seconden voor timeout)",
    )

    # Doel configuratie
    escaleer_naar: EscalatieDoelType = Field(
        ...,
        description="Naar wie wordt geëscaleerd",
    )
    doel_id: UUID | None = Field(
        default=None,
        description="Specifiek doel ID (voor SPECIFIEK type)",
    )

    # Actie configuratie
    actie: EscalatieActieType = Field(
        default=EscalatieActieType.MELDEN,
        description="Welke actie wordt ondernomen bij escalatie",
    )

    # Scope
    van_toepassing_op: Literal["alle", "afdeling", "crew", "agent"] = Field(
        default="alle",
        description="Scope van deze regel",
    )
    scope_id: UUID | None = Field(
        default=None,
        description="ID van specifieke scope (afdeling/crew/agent)",
    )

    # Prioriteit en status
    prioriteit: int = Field(
        default=0,
        description="Prioriteit (hoger = belangrijker)",
    )
    actief: bool = Field(
        default=True,
        description="Of deze regel actief is",
    )

    # Metadata
    aangemaakt_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van aanmaken",
    )

    def is_van_toepassing(
        self,
        afdeling_id: UUID | None = None,
        crew_id: UUID | None = None,
        agent_id: UUID | None = None,
    ) -> bool:
        """Check of deze regel van toepassing is op gegeven scope.

        Args:
            afdeling_id: Optioneel afdeling ID.
            crew_id: Optioneel crew ID.
            agent_id: Optioneel agent ID.

        Returns:
            True als de regel van toepassing is.
        """
        if not self.actief:
            return False

        if self.van_toepassing_op == "alle":
            return True

        if self.van_toepassing_op == "afdeling" and afdeling_id:
            return self.scope_id == afdeling_id

        if self.van_toepassing_op == "crew" and crew_id:
            return self.scope_id == crew_id

        if self.van_toepassing_op == "agent" and agent_id:
            return self.scope_id == agent_id

        return False


class Escalatie(BaseModel):
    """Een actieve of voltooide escalatie.

    Representeert een specifiek escalatie-event met alle
    bijbehorende informatie en status.
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor deze escalatie",
    )

    # Bron
    bron_id: UUID = Field(
        ...,
        description="ID van de entiteit die escaleert (agent/crew)",
    )
    bron_type: Literal["agent", "crew"] = Field(
        ...,
        description="Type bron entiteit",
    )

    # Regel die triggerde
    regel_id: UUID | None = Field(
        default=None,
        description="ID van de regel die deze escalatie triggerde",
    )

    # Doel
    doel_id: UUID = Field(
        ...,
        description="ID van de entiteit waarnaar geëscaleerd wordt",
    )
    doel_type: Literal["agent", "crew", "afdeling"] = Field(
        ...,
        description="Type doel entiteit",
    )

    # Details
    reden: str = Field(
        ...,
        description="Reden voor de escalatie",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context informatie",
    )
    trigger_type: EscalatieTriggerType = Field(
        ...,
        description="Type trigger dat de escalatie veroorzaakte",
    )
    actie: EscalatieActieType = Field(
        ...,
        description="Actie die ondernomen wordt/is",
    )

    # Status
    status: EscalatieStatus = Field(
        default=EscalatieStatus.NIEUW,
        description="Huidige status van de escalatie",
    )
    prioriteit: int = Field(
        default=0,
        description="Prioriteit van deze escalatie",
    )

    # Tracking
    aangemaakt_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van aanmaken",
    )
    behandeld_op: datetime | None = Field(
        default=None,
        description="Tijdstip van behandeling",
    )
    opgelost_op: datetime | None = Field(
        default=None,
        description="Tijdstip van oplossing",
    )

    # Reactie
    reactie: str | None = Field(
        default=None,
        description="Reactie van het escalatiedoel",
    )
    behandeld_door: UUID | None = Field(
        default=None,
        description="ID van de entiteit die de escalatie behandelde",
    )

    def markeer_in_behandeling(self, behandeld_door: UUID) -> None:
        """Markeer de escalatie als in behandeling.

        Args:
            behandeld_door: ID van de behandelaar.
        """
        self.status = EscalatieStatus.IN_BEHANDELING
        self.behandeld_op = datetime.now(timezone.utc)
        self.behandeld_door = behandeld_door

    def markeer_opgelost(self, reactie: str) -> None:
        """Markeer de escalatie als opgelost.

        Args:
            reactie: De reactie/oplossing.
        """
        self.status = EscalatieStatus.OPGELOST
        self.opgelost_op = datetime.now(timezone.utc)
        self.reactie = reactie

    def markeer_doorgestuurd(self, nieuw_doel_id: UUID) -> None:
        """Markeer de escalatie als doorgestuurd.

        Args:
            nieuw_doel_id: ID van het nieuwe escalatiedoel.
        """
        self.status = EscalatieStatus.DOORGESTUURD
        self.context["oorspronkelijk_doel_id"] = str(self.doel_id)
        self.doel_id = nieuw_doel_id

    def is_open(self) -> bool:
        """Check of de escalatie nog open is.

        Returns:
            True als nog open.
        """
        return self.status in [EscalatieStatus.NIEUW, EscalatieStatus.IN_BEHANDELING]


class EscalatieManager(BaseModel):
    """Beheert escalaties en regels.

    Deze klasse coördineert het escalatieproces, controleert
    triggers, en beheert actieve escalaties.

    Voorbeeld:
        ```python
        from crewai.governance import EscalatieManager, EscalatieRegel

        manager = EscalatieManager()

        # Voeg regels toe
        manager.voeg_regel_toe(EscalatieRegel(
            naam="Budget escalatie",
            trigger_type=EscalatieTriggerType.BUDGET_OVERSCHREDEN,
            trigger_waarde=1000.0,
            escaleer_naar=EscalatieDoelType.AFDELING_HOOFD,
            actie=EscalatieActieType.GOEDKEURING_VRAGEN
        ))

        # Controleer triggers
        triggered = manager.controleer_triggers(
            context={"budget_gebruikt": 1500.0}
        )

        # Maak escalatie
        if triggered:
            escalatie = manager.escaleer(
                bron_id=agent_id,
                bron_type="agent",
                regel=triggered[0],
                reden="Budget limiet bereikt"
            )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    regels: list[EscalatieRegel] = Field(
        default_factory=list,
        description="Alle escalatieregels",
    )
    actieve_escalaties: dict[UUID, Escalatie] = Field(
        default_factory=dict,
        description="Alle actieve escalaties",
    )
    escalatie_geschiedenis: list[Escalatie] = Field(
        default_factory=list,
        description="Geschiedenis van voltooide escalaties",
    )

    # Organisatie referentie voor het opzoeken van management keten
    organisatie: "OrganisatieHierarchie | None" = Field(
        default=None,
        description="Referentie naar organisatie voor management keten",
    )

    # Callback voor escalatie notificaties
    on_escalatie: Callable[[Escalatie], None] | None = Field(
        default=None,
        description="Callback functie bij nieuwe escalatie",
    )

    def voeg_regel_toe(self, regel: EscalatieRegel) -> None:
        """Voeg een escalatieregel toe.

        Args:
            regel: De regel om toe te voegen.
        """
        self.regels.append(regel)
        # Sorteer op prioriteit
        self.regels.sort(key=lambda r: r.prioriteit, reverse=True)

    def verwijder_regel(self, regel_id: UUID) -> bool:
        """Verwijder een escalatieregel.

        Args:
            regel_id: ID van de te verwijderen regel.

        Returns:
            True als verwijderd.
        """
        for i, regel in enumerate(self.regels):
            if regel.id == regel_id:
                self.regels.pop(i)
                return True
        return False

    def krijg_regel(self, regel_id: UUID) -> EscalatieRegel | None:
        """Krijg een regel op ID.

        Args:
            regel_id: ID van de regel.

        Returns:
            De regel of None.
        """
        for regel in self.regels:
            if regel.id == regel_id:
                return regel
        return None

    def controleer_triggers(
        self,
        context: dict[str, Any],
        afdeling_id: UUID | None = None,
        crew_id: UUID | None = None,
        agent_id: UUID | None = None,
    ) -> list[EscalatieRegel]:
        """Controleer welke escalatieregels triggeren.

        Args:
            context: Context met data voor trigger evaluatie.
            afdeling_id: Optioneel afdeling ID.
            crew_id: Optioneel crew ID.
            agent_id: Optioneel agent ID.

        Returns:
            Lijst van getriggerde regels.
        """
        getriggerd: list[EscalatieRegel] = []

        for regel in self.regels:
            if not regel.is_van_toepassing(afdeling_id, crew_id, agent_id):
                continue

            if self._evalueer_trigger(regel, context):
                getriggerd.append(regel)

        return getriggerd

    def _evalueer_trigger(
        self, regel: EscalatieRegel, context: dict[str, Any]
    ) -> bool:
        """Evalueer of een trigger regel voldoet.

        Args:
            regel: De regel om te evalueren.
            context: Context met data.

        Returns:
            True als de trigger voldoet.
        """
        if regel.trigger_type == EscalatieTriggerType.TIMEOUT:
            # Check of tijd verstreken is
            start_tijd = context.get("start_tijd")
            if start_tijd is None:
                return False
            if isinstance(start_tijd, str):
                start_tijd = datetime.fromisoformat(start_tijd)
            verstreken = (datetime.now(timezone.utc) - start_tijd).total_seconds()
            return verstreken > regel.trigger_waarde

        elif regel.trigger_type == EscalatieTriggerType.FOUT:
            # Check of er een fout is
            return context.get("fout", False)

        elif regel.trigger_type == EscalatieTriggerType.BUDGET_OVERSCHREDEN:
            # Check budget
            budget_gebruikt = context.get("budget_gebruikt", 0.0)
            return budget_gebruikt > regel.trigger_waarde

        elif regel.trigger_type == EscalatieTriggerType.HERHAALDE_POGINGEN:
            # Check aantal pogingen
            pogingen = context.get("pogingen", 0)
            return pogingen >= regel.trigger_waarde

        elif regel.trigger_type == EscalatieTriggerType.GEEN_VOORTGANG:
            # Check laatste activiteit
            laatste_activiteit = context.get("laatste_activiteit")
            if laatste_activiteit is None:
                return False
            if isinstance(laatste_activiteit, str):
                laatste_activiteit = datetime.fromisoformat(laatste_activiteit)
            inactief = (datetime.now(timezone.utc) - laatste_activiteit).total_seconds()
            return inactief > regel.trigger_waarde

        elif regel.trigger_type == EscalatieTriggerType.HANDMATIG:
            # Altijd handmatig getriggerd
            return context.get("handmatige_escalatie", False)

        return False

    def escaleer(
        self,
        bron_id: UUID,
        bron_type: Literal["agent", "crew"],
        regel: EscalatieRegel | None,
        reden: str,
        context: dict[str, Any] | None = None,
    ) -> Escalatie:
        """Maak een nieuwe escalatie.

        Args:
            bron_id: ID van de bron entiteit.
            bron_type: Type bron.
            regel: Optionele regel die triggerde.
            reden: Reden voor escalatie.
            context: Extra context.

        Returns:
            De nieuwe escalatie.
        """
        if context is None:
            context = {}

        # Bepaal doel
        doel_id, doel_type = self._bepaal_escalatie_doel(
            bron_id=bron_id,
            bron_type=bron_type,
            regel=regel,
        )

        # Maak escalatie
        escalatie = Escalatie(
            bron_id=bron_id,
            bron_type=bron_type,
            regel_id=regel.id if regel else None,
            doel_id=doel_id,
            doel_type=doel_type,
            reden=reden,
            context=context,
            trigger_type=regel.trigger_type if regel else EscalatieTriggerType.HANDMATIG,
            actie=regel.actie if regel else EscalatieActieType.MELDEN,
            prioriteit=regel.prioriteit if regel else 0,
        )

        # Registreer
        self.actieve_escalaties[escalatie.id] = escalatie

        # Trigger callback
        if self.on_escalatie:
            self.on_escalatie(escalatie)

        return escalatie

    def _bepaal_escalatie_doel(
        self,
        bron_id: UUID,
        bron_type: str,
        regel: EscalatieRegel | None,
    ) -> tuple[UUID, Literal["agent", "crew", "afdeling"]]:
        """Bepaal het doel van de escalatie.

        Args:
            bron_id: ID van de bron.
            bron_type: Type bron.
            regel: Optionele regel.

        Returns:
            Tuple van (doel_id, doel_type).
        """
        if regel is None:
            # Zonder regel, probeer directe manager
            if self.organisatie:
                manager = self.organisatie.krijg_directe_manager(bron_id)
                if manager:
                    return manager, "agent"
            # Fallback
            return bron_id, bron_type

        if regel.escaleer_naar == EscalatieDoelType.SPECIFIEK:
            if regel.doel_id:
                return regel.doel_id, "agent"  # Assumeer agent

        if self.organisatie is None:
            # Zonder organisatie, return bron als fallback
            return bron_id, bron_type

        if regel.escaleer_naar == EscalatieDoelType.DIRECTE_MANAGER:
            manager = self.organisatie.krijg_directe_manager(bron_id)
            if manager:
                return manager, "agent"

        elif regel.escaleer_naar == EscalatieDoelType.VOLGENDE_IN_KETEN:
            keten = self.organisatie.krijg_management_keten(bron_id)
            if keten:
                return keten[0], "agent"

        elif regel.escaleer_naar == EscalatieDoelType.AFDELING_HOOFD:
            afdeling = self.organisatie.afdelingen_manager.krijg_afdeling_voor_agent(
                bron_id
            )
            if afdeling and afdeling.manager_agent_id:
                return afdeling.manager_agent_id, "agent"

        elif regel.escaleer_naar == EscalatieDoelType.DIRECTIE:
            # Zoek hoogste in keten
            keten = self.organisatie.krijg_management_keten(bron_id)
            if keten:
                return keten[-1], "agent"

        # Fallback naar bron
        return bron_id, bron_type

    def behandel_escalatie(
        self,
        escalatie_id: UUID,
        behandeld_door: UUID,
    ) -> Escalatie | None:
        """Markeer een escalatie als in behandeling.

        Args:
            escalatie_id: ID van de escalatie.
            behandeld_door: ID van de behandelaar.

        Returns:
            De bijgewerkte escalatie of None.
        """
        escalatie = self.actieve_escalaties.get(escalatie_id)
        if escalatie is None:
            return None

        escalatie.markeer_in_behandeling(behandeld_door)
        return escalatie

    def los_escalatie_op(
        self,
        escalatie_id: UUID,
        reactie: str,
    ) -> Escalatie | None:
        """Los een escalatie op.

        Args:
            escalatie_id: ID van de escalatie.
            reactie: De reactie/oplossing.

        Returns:
            De opgeloste escalatie of None.
        """
        escalatie = self.actieve_escalaties.get(escalatie_id)
        if escalatie is None:
            return None

        escalatie.markeer_opgelost(reactie)

        # Verplaats naar geschiedenis
        del self.actieve_escalaties[escalatie_id]
        self.escalatie_geschiedenis.append(escalatie)

        return escalatie

    def stuur_door(
        self,
        escalatie_id: UUID,
        nieuw_doel_id: UUID,
    ) -> Escalatie | None:
        """Stuur een escalatie door naar een ander doel.

        Args:
            escalatie_id: ID van de escalatie.
            nieuw_doel_id: ID van het nieuwe doel.

        Returns:
            De bijgewerkte escalatie of None.
        """
        escalatie = self.actieve_escalaties.get(escalatie_id)
        if escalatie is None:
            return None

        escalatie.markeer_doorgestuurd(nieuw_doel_id)
        return escalatie

    def krijg_openstaande_escalaties(
        self, doel_id: UUID | None = None
    ) -> list[Escalatie]:
        """Krijg alle openstaande escalaties.

        Args:
            doel_id: Optioneel filter op doel ID.

        Returns:
            Lijst van openstaande escalaties.
        """
        escalaties = [e for e in self.actieve_escalaties.values() if e.is_open()]

        if doel_id is not None:
            escalaties = [e for e in escalaties if e.doel_id == doel_id]

        return sorted(escalaties, key=lambda e: e.prioriteit, reverse=True)

    def krijg_escalatie_statistieken(self) -> dict[str, Any]:
        """Krijg statistieken over escalaties.

        Returns:
            Dictionary met statistieken.
        """
        alle_escalaties = list(self.actieve_escalaties.values()) + list(
            self.escalatie_geschiedenis
        )

        if not alle_escalaties:
            return {
                "totaal": 0,
                "actief": 0,
                "opgelost": 0,
                "gemiddelde_oplostijd_seconden": 0,
            }

        opgeloste = [
            e for e in alle_escalaties if e.status == EscalatieStatus.OPGELOST
        ]
        oplostijden = []
        for e in opgeloste:
            if e.opgelost_op and e.aangemaakt_op:
                oplostijden.append(
                    (e.opgelost_op - e.aangemaakt_op).total_seconds()
                )

        return {
            "totaal": len(alle_escalaties),
            "actief": len(self.actieve_escalaties),
            "opgelost": len(opgeloste),
            "gemiddelde_oplostijd_seconden": (
                sum(oplostijden) / len(oplostijden) if oplostijden else 0
            ),
            "per_trigger_type": {
                trigger.value: len(
                    [e for e in alle_escalaties if e.trigger_type == trigger]
                )
                for trigger in EscalatieTriggerType
            },
        }


# Voorgedefinieerde escalatieregels
def maak_standaard_escalatie_regels() -> list[EscalatieRegel]:
    """Maak een set standaard escalatieregels.

    Returns:
        Lijst met standaard regels.
    """
    return [
        EscalatieRegel(
            naam="Timeout escalatie (1 uur)",
            beschrijving="Escaleer naar manager als taak langer dan 1 uur duurt",
            trigger_type=EscalatieTriggerType.TIMEOUT,
            trigger_waarde=3600,  # 1 uur
            escaleer_naar=EscalatieDoelType.DIRECTE_MANAGER,
            actie=EscalatieActieType.MELDEN,
            prioriteit=1,
        ),
        EscalatieRegel(
            naam="Fout escalatie",
            beschrijving="Escaleer naar manager bij kritieke fouten",
            trigger_type=EscalatieTriggerType.FOUT,
            escaleer_naar=EscalatieDoelType.DIRECTE_MANAGER,
            actie=EscalatieActieType.MELDEN,
            prioriteit=5,
        ),
        EscalatieRegel(
            naam="Budget overschrijding",
            beschrijving="Escaleer naar afdelingshoofd bij budget overschrijding",
            trigger_type=EscalatieTriggerType.BUDGET_OVERSCHREDEN,
            trigger_waarde=1000.0,
            escaleer_naar=EscalatieDoelType.AFDELING_HOOFD,
            actie=EscalatieActieType.GOEDKEURING_VRAGEN,
            prioriteit=10,
        ),
        EscalatieRegel(
            naam="Herhaalde mislukkingen",
            beschrijving="Escaleer na 3 mislukte pogingen",
            trigger_type=EscalatieTriggerType.HERHAALDE_POGINGEN,
            trigger_waarde=3,
            escaleer_naar=EscalatieDoelType.DIRECTE_MANAGER,
            actie=EscalatieActieType.HERTOEWIJZEN,
            prioriteit=3,
        ),
        EscalatieRegel(
            naam="Geen voortgang",
            beschrijving="Escaleer als 30 minuten geen activiteit",
            trigger_type=EscalatieTriggerType.GEEN_VOORTGANG,
            trigger_waarde=1800,  # 30 minuten
            escaleer_naar=EscalatieDoelType.DIRECTE_MANAGER,
            actie=EscalatieActieType.MELDEN,
            prioriteit=2,
        ),
    ]
