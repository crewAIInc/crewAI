"""Toegangscontrole systeem voor enterprise governance.

Dit module implementeert een flexibel toegangscontrolesysteem
waarmee developers kunnen configureren wie wat mag doen.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any, Callable, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


ResourceType = Literal["crew", "agent", "tool", "geheugen", "rapport", "opdracht"]
"""Type resource waarvoor toegang gecontroleerd wordt."""

PrincipalType = Literal["agent", "rol", "afdeling", "crew"]
"""Type entiteit die toegang vraagt."""

ActieType = Literal[
    "lezen", "schrijven", "uitvoeren", "delegeren", "goedkeuren", "escaleren"
]
"""Type actie dat uitgevoerd wordt."""

EffectType = Literal["toestaan", "weigeren"]
"""Of toegang wordt toegestaan of geweigerd."""


class TijdsVoorwaarde(BaseModel):
    """Voorwaarde gebaseerd op tijd.

    Gebruikt om toegang te beperken tot bepaalde tijden.
    """

    start_tijd: time = Field(
        default=time(0, 0),
        description="Starttijd van toegestane periode",
    )
    eind_tijd: time = Field(
        default=time(23, 59, 59),
        description="Eindtijd van toegestane periode",
    )
    weekdagen: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6],
        description="Toegestane weekdagen (0=maandag, 6=zondag)",
    )

    def is_binnen_periode(self, dt: datetime | None = None) -> bool:
        """Check of huidige tijd binnen de toegestane periode valt.

        Args:
            dt: Optionele datetime om te checken (anders nu).

        Returns:
            True als binnen periode.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)

        # Check weekdag
        if dt.weekday() not in self.weekdagen:
            return False

        # Check tijd
        huidige_tijd = dt.time()
        if self.start_tijd <= self.eind_tijd:
            return self.start_tijd <= huidige_tijd <= self.eind_tijd
        else:
            # Over middernacht
            return huidige_tijd >= self.start_tijd or huidige_tijd <= self.eind_tijd


class BudgetVoorwaarde(BaseModel):
    """Voorwaarde gebaseerd op budget.

    Gebruikt om acties te beperken tot een maximaal bedrag.
    """

    max_bedrag: float = Field(
        ...,
        description="Maximum bedrag per actie",
    )
    max_totaal: float | None = Field(
        default=None,
        description="Maximum totaal budget (None = onbeperkt)",
    )
    totaal_gebruikt: float = Field(
        default=0.0,
        description="Totaal verbruikt budget",
    )

    def is_binnen_budget(self, bedrag: float) -> bool:
        """Check of een bedrag binnen budget valt.

        Args:
            bedrag: Het te besteden bedrag.

        Returns:
            True als binnen budget.
        """
        if bedrag > self.max_bedrag:
            return False
        if self.max_totaal is not None:
            if (self.totaal_gebruikt + bedrag) > self.max_totaal:
                return False
        return True

    def registreer_uitgave(self, bedrag: float) -> bool:
        """Registreer een uitgave.

        Args:
            bedrag: Het uitgegeven bedrag.

        Returns:
            True als geregistreerd, False als budget overschreden.
        """
        if not self.is_binnen_budget(bedrag):
            return False
        self.totaal_gebruikt += bedrag
        return True


class ToegangsVoorwaarden(BaseModel):
    """Verzameling voorwaarden voor een toegangsregel.

    Alle voorwaarden moeten voldaan zijn voor toegang.
    """

    tijd: TijdsVoorwaarde | None = Field(
        default=None,
        description="Tijdsgebonden voorwaarde",
    )
    budget: BudgetVoorwaarde | None = Field(
        default=None,
        description="Budgetvoorwaarde",
    )
    vereist_goedkeuring: bool = Field(
        default=False,
        description="Of goedkeuring vereist is",
    )
    max_per_dag: int | None = Field(
        default=None,
        description="Maximum aantal acties per dag",
    )
    custom_voorwaarden: dict[str, Any] = Field(
        default_factory=dict,
        description="Aangepaste voorwaarden",
    )

    def zijn_voldaan(self, context: dict[str, Any] | None = None) -> tuple[bool, str]:
        """Check of alle voorwaarden voldaan zijn.

        Args:
            context: Optionele context met extra informatie.

        Returns:
            Tuple van (voldaan, reden als niet voldaan).
        """
        if context is None:
            context = {}

        # Check tijdsvoorwaarde
        if self.tijd is not None:
            if not self.tijd.is_binnen_periode():
                return False, "Actie niet toegestaan op dit tijdstip"

        # Check budgetvoorwaarde
        if self.budget is not None:
            bedrag = context.get("bedrag", 0.0)
            if not self.budget.is_binnen_budget(bedrag):
                return False, "Budget overschreden"

        # Check goedkeuringsvereiste
        if self.vereist_goedkeuring:
            if not context.get("goedgekeurd", False):
                return False, "Goedkeuring vereist"

        return True, ""


class ToegangsRegel(BaseModel):
    """Enkele toegangsregel in het toegangscontrolesysteem.

    Een regel definieert of een specifieke principal een bepaalde
    actie mag uitvoeren op een resource.

    Voorbeeld:
        ```python
        from crewai.governance import ToegangsRegel

        # Regel: Trading agents mogen trade tools gebruiken
        regel = ToegangsRegel(
            resource_type="tool",
            resource_id=trade_tool_id,
            principal_type="afdeling",
            principal_id=trading_afdeling_id,
            actie="uitvoeren",
            effect="toestaan"
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor deze regel",
    )
    naam: str | None = Field(
        default=None,
        description="Optionele naam voor de regel",
    )
    beschrijving: str | None = Field(
        default=None,
        description="Optionele beschrijving",
    )

    # Wat
    resource_type: ResourceType = Field(
        ...,
        description="Type resource waar de regel op van toepassing is",
    )
    resource_id: UUID | None = Field(
        default=None,
        description="Specifieke resource ID (None = alle van dit type)",
    )

    # Wie
    principal_type: PrincipalType = Field(
        ...,
        description="Type entiteit die toegang vraagt",
    )
    principal_id: UUID = Field(
        ...,
        description="ID van de principal",
    )

    # Actie
    actie: ActieType = Field(
        ...,
        description="De actie waarop deze regel van toepassing is",
    )

    # Effect
    effect: EffectType = Field(
        default="toestaan",
        description="Of toegang wordt toegestaan of geweigerd",
    )

    # Voorwaarden
    voorwaarden: ToegangsVoorwaarden = Field(
        default_factory=ToegangsVoorwaarden,
        description="Voorwaarden die moeten gelden",
    )

    # Prioriteit (hogere prioriteit wint bij conflict)
    prioriteit: int = Field(
        default=0,
        description="Prioriteit van de regel (hoger = belangrijker)",
    )

    # Status
    actief: bool = Field(
        default=True,
        description="Of deze regel actief is",
    )

    # Tracking
    aangemaakt_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van aanmaken",
    )

    def is_van_toepassing(
        self,
        principal_id: UUID,
        principal_type: PrincipalType,
        resource_id: UUID | None,
        resource_type: ResourceType,
        actie: ActieType,
    ) -> bool:
        """Check of deze regel van toepassing is op een verzoek.

        Args:
            principal_id: ID van de vragende entiteit.
            principal_type: Type van de vragende entiteit.
            resource_id: ID van de resource (kan None zijn).
            resource_type: Type van de resource.
            actie: De gevraagde actie.

        Returns:
            True als de regel van toepassing is.
        """
        if not self.actief:
            return False

        # Check principal
        if self.principal_type != principal_type:
            return False
        if self.principal_id != principal_id:
            return False

        # Check resource type
        if self.resource_type != resource_type:
            return False

        # Check resource ID (None = alle)
        if self.resource_id is not None and resource_id is not None:
            if self.resource_id != resource_id:
                return False

        # Check actie
        if self.actie != actie:
            return False

        return True

    def evalueer(self, context: dict[str, Any] | None = None) -> tuple[bool, str]:
        """Evalueer de regel met voorwaarden.

        Args:
            context: Optionele context voor voorwaarden.

        Returns:
            Tuple van (toegestaan, reden).
        """
        # Check voorwaarden
        voldaan, reden = self.voorwaarden.zijn_voldaan(context)
        if not voldaan:
            return False, reden

        # Return effect
        if self.effect == "toestaan":
            return True, "Toegang toegestaan door regel"
        else:
            return False, "Toegang geweigerd door regel"


class ToegangsControle(BaseModel):
    """Beheert alle toegangsregels en controleert toegang.

    Deze klasse is het centrale punt voor toegangscontrole.
    Het evalueert regels op basis van prioriteit en geeft
    een beslissing terug.

    Voorbeeld:
        ```python
        from crewai.governance import ToegangsControle, ToegangsRegel

        controle = ToegangsControle()

        # Voeg regels toe
        controle.voeg_regel_toe(ToegangsRegel(
            resource_type="tool",
            principal_type="agent",
            principal_id=agent_id,
            actie="uitvoeren",
            effect="toestaan"
        ))

        # Controleer toegang
        toegestaan, reden = controle.controleer_toegang(
            principal_id=agent_id,
            principal_type="agent",
            resource_id=tool_id,
            resource_type="tool",
            actie="uitvoeren"
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    regels: list[ToegangsRegel] = Field(
        default_factory=list,
        description="Alle toegangsregels",
    )

    # Standaard beleid als geen regels matchen
    standaard_effect: EffectType = Field(
        default="weigeren",
        description="Standaard effect als geen regels matchen",
    )

    # Audit logging
    log_toegang: bool = Field(
        default=True,
        description="Of toegangscontroles gelogd worden",
    )
    toegangs_log: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Log van toegangscontroles",
    )

    def voeg_regel_toe(self, regel: ToegangsRegel) -> None:
        """Voeg een toegangsregel toe.

        Args:
            regel: De regel om toe te voegen.
        """
        self.regels.append(regel)
        # Sorteer op prioriteit (hoogste eerst)
        self.regels.sort(key=lambda r: r.prioriteit, reverse=True)

    def verwijder_regel(self, regel_id: UUID) -> bool:
        """Verwijder een toegangsregel.

        Args:
            regel_id: ID van de te verwijderen regel.

        Returns:
            True als verwijderd, False als niet gevonden.
        """
        for i, regel in enumerate(self.regels):
            if regel.id == regel_id:
                self.regels.pop(i)
                return True
        return False

    def krijg_regel(self, regel_id: UUID) -> ToegangsRegel | None:
        """Krijg een regel op ID.

        Args:
            regel_id: ID van de regel.

        Returns:
            De regel of None als niet gevonden.
        """
        for regel in self.regels:
            if regel.id == regel_id:
                return regel
        return None

    def controleer_toegang(
        self,
        principal_id: UUID,
        principal_type: PrincipalType,
        resource_id: UUID | None,
        resource_type: ResourceType,
        actie: ActieType,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Controleer of een actie is toegestaan.

        Evalueert alle toepasselijke regels op volgorde van prioriteit.
        De eerste matchende regel bepaalt het resultaat.

        Args:
            principal_id: ID van de vragende entiteit.
            principal_type: Type van de vragende entiteit.
            resource_id: ID van de resource (kan None zijn).
            resource_type: Type van de resource.
            actie: De gevraagde actie.
            context: Optionele context voor voorwaarden.

        Returns:
            Tuple van (toegestaan, reden).
        """
        if context is None:
            context = {}

        # Zoek toepasselijke regels (gesorteerd op prioriteit)
        for regel in self.regels:
            if regel.is_van_toepassing(
                principal_id=principal_id,
                principal_type=principal_type,
                resource_id=resource_id,
                resource_type=resource_type,
                actie=actie,
            ):
                toegestaan, reden = regel.evalueer(context)

                # Log de beslissing
                if self.log_toegang:
                    self._log_beslissing(
                        principal_id=principal_id,
                        resource_id=resource_id,
                        actie=actie,
                        toegestaan=toegestaan,
                        reden=reden,
                        regel_id=regel.id,
                    )

                return toegestaan, reden

        # Geen regels gevonden, gebruik standaard
        toegestaan = self.standaard_effect == "toestaan"
        reden = f"Geen regels gevonden, standaard: {self.standaard_effect}"

        if self.log_toegang:
            self._log_beslissing(
                principal_id=principal_id,
                resource_id=resource_id,
                actie=actie,
                toegestaan=toegestaan,
                reden=reden,
                regel_id=None,
            )

        return toegestaan, reden

    def krijg_toegankelijke_resources(
        self,
        principal_id: UUID,
        principal_type: PrincipalType,
        resource_type: ResourceType,
        actie: ActieType,
    ) -> list[UUID]:
        """Krijg alle resources waartoe een principal toegang heeft.

        Args:
            principal_id: ID van de principal.
            principal_type: Type van de principal.
            resource_type: Type resource om te controleren.
            actie: De actie waarvoor toegang gecontroleerd wordt.

        Returns:
            Lijst van resource IDs waartoe toegang is.
        """
        toegankelijk: list[UUID] = []

        for regel in self.regels:
            if not regel.actief:
                continue
            if regel.principal_type != principal_type:
                continue
            if regel.principal_id != principal_id:
                continue
            if regel.resource_type != resource_type:
                continue
            if regel.actie != actie:
                continue
            if regel.effect != "toestaan":
                continue

            if regel.resource_id is not None:
                if regel.resource_id not in toegankelijk:
                    toegankelijk.append(regel.resource_id)

        return toegankelijk

    def _log_beslissing(
        self,
        principal_id: UUID,
        resource_id: UUID | None,
        actie: ActieType,
        toegestaan: bool,
        reden: str,
        regel_id: UUID | None,
    ) -> None:
        """Log een toegangsbeslissing.

        Args:
            principal_id: ID van de principal.
            resource_id: ID van de resource.
            actie: De gevraagde actie.
            toegestaan: Of toegang is verleend.
            reden: Reden voor de beslissing.
            regel_id: ID van de bepalende regel.
        """
        self.toegangs_log.append(
            {
                "tijdstip": datetime.now(timezone.utc).isoformat(),
                "principal_id": str(principal_id),
                "resource_id": str(resource_id) if resource_id else None,
                "actie": actie,
                "toegestaan": toegestaan,
                "reden": reden,
                "regel_id": str(regel_id) if regel_id else None,
            }
        )

        # Beperk log grootte
        max_log_entries = 1000
        if len(self.toegangs_log) > max_log_entries:
            self.toegangs_log = self.toegangs_log[-max_log_entries:]

    def krijg_regels_voor_principal(
        self, principal_id: UUID, principal_type: PrincipalType
    ) -> list[ToegangsRegel]:
        """Krijg alle regels voor een specifieke principal.

        Args:
            principal_id: ID van de principal.
            principal_type: Type van de principal.

        Returns:
            Lijst van toepasselijke regels.
        """
        return [
            regel
            for regel in self.regels
            if regel.principal_id == principal_id
            and regel.principal_type == principal_type
            and regel.actief
        ]

    def krijg_regels_voor_resource(
        self, resource_id: UUID, resource_type: ResourceType
    ) -> list[ToegangsRegel]:
        """Krijg alle regels voor een specifieke resource.

        Args:
            resource_id: ID van de resource.
            resource_type: Type van de resource.

        Returns:
            Lijst van toepasselijke regels.
        """
        return [
            regel
            for regel in self.regels
            if (regel.resource_id == resource_id or regel.resource_id is None)
            and regel.resource_type == resource_type
            and regel.actief
        ]


# Helper functies voor veelvoorkomende regels
def maak_allow_all_regel(
    principal_id: UUID,
    principal_type: PrincipalType,
    resource_type: ResourceType,
    naam: str | None = None,
) -> ToegangsRegel:
    """Maak een regel die alle acties toestaat.

    Args:
        principal_id: ID van de principal.
        principal_type: Type van de principal.
        resource_type: Type resource.
        naam: Optionele naam voor de regel.

    Returns:
        De toegangsregel.
    """
    return ToegangsRegel(
        naam=naam or f"Allow all {resource_type}",
        resource_type=resource_type,
        resource_id=None,  # Alle resources
        principal_type=principal_type,
        principal_id=principal_id,
        actie="uitvoeren",
        effect="toestaan",
        prioriteit=0,
    )


def maak_werkuren_regel(
    principal_id: UUID,
    principal_type: PrincipalType,
    resource_type: ResourceType,
    actie: ActieType,
    start_uur: int = 9,
    eind_uur: int = 17,
    naam: str | None = None,
) -> ToegangsRegel:
    """Maak een regel beperkt tot werkuren.

    Args:
        principal_id: ID van de principal.
        principal_type: Type van de principal.
        resource_type: Type resource.
        actie: De actie.
        start_uur: Startuur (default 9).
        eind_uur: Einduur (default 17).
        naam: Optionele naam.

    Returns:
        De toegangsregel.
    """
    return ToegangsRegel(
        naam=naam or f"Werkuren {resource_type}",
        resource_type=resource_type,
        resource_id=None,
        principal_type=principal_type,
        principal_id=principal_id,
        actie=actie,
        effect="toestaan",
        voorwaarden=ToegangsVoorwaarden(
            tijd=TijdsVoorwaarde(
                start_tijd=time(start_uur, 0),
                eind_tijd=time(eind_uur, 0),
                weekdagen=[0, 1, 2, 3, 4],  # Ma-Vr
            )
        ),
    )


def maak_budget_regel(
    principal_id: UUID,
    principal_type: PrincipalType,
    resource_type: ResourceType,
    actie: ActieType,
    max_bedrag: float,
    max_totaal: float | None = None,
    naam: str | None = None,
) -> ToegangsRegel:
    """Maak een regel met budgetlimiet.

    Args:
        principal_id: ID van de principal.
        principal_type: Type van de principal.
        resource_type: Type resource.
        actie: De actie.
        max_bedrag: Maximum per actie.
        max_totaal: Maximum totaal budget.
        naam: Optionele naam.

    Returns:
        De toegangsregel.
    """
    return ToegangsRegel(
        naam=naam or f"Budget {resource_type}",
        resource_type=resource_type,
        resource_id=None,
        principal_type=principal_type,
        principal_id=principal_id,
        actie=actie,
        effect="toestaan",
        voorwaarden=ToegangsVoorwaarden(
            budget=BudgetVoorwaarde(
                max_bedrag=max_bedrag,
                max_totaal=max_totaal,
            )
        ),
    )
