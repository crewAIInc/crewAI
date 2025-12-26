"""Rapportages van crews naar management.

Dit module implementeert het rapportagesysteem waarmee crews
en agents kunnen rapporteren aan hun management.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from crewai.organization.hierarchy import OrganisatieHierarchie


class RapportType(str, Enum):
    """Type rapport."""

    STATUS = "status"
    """Statusupdate over lopende werkzaamheden."""

    PROBLEEM = "probleem"
    """Melding van een probleem of blokkade."""

    RESULTAAT = "resultaat"
    """Rapportage van behaald resultaat."""

    ESCALATIE = "escalatie"
    """Escalatierapport naar hoger niveau."""

    VOORTGANG = "voortgang"
    """Periodieke voortgangsrapportage."""

    ANALYSE = "analyse"
    """Analyserapport met bevindingen."""

    AANBEVELING = "aanbeveling"
    """Rapport met aanbevelingen voor actie."""


class RapportPrioriteit(str, Enum):
    """Prioriteit van een rapport."""

    INFO = "info"
    """Informatief, geen actie vereist."""

    NORMAAL = "normaal"
    """Normale prioriteit."""

    BELANGRIJK = "belangrijk"
    """Belangrijk, aandacht vereist."""

    URGENT = "urgent"
    """Urgent, directe aandacht vereist."""


class RapportBijlage(BaseModel):
    """Bijlage bij een rapport."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor de bijlage",
    )
    naam: str = Field(
        ...,
        description="Naam van de bijlage",
    )
    type: str = Field(
        ...,
        description="Type bijlage (bijv. 'json', 'tekst', 'grafiek')",
    )
    inhoud: Any = Field(
        ...,
        description="Inhoud van de bijlage",
    )
    beschrijving: str | None = Field(
        default=None,
        description="Optionele beschrijving",
    )


class RapportReactie(BaseModel):
    """Reactie op een rapport."""

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor de reactie",
    )
    auteur_id: UUID = Field(
        ...,
        description="ID van de auteur van de reactie",
    )
    tijdstip: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van de reactie",
    )
    inhoud: str = Field(
        ...,
        description="Inhoud van de reactie",
    )
    actie_vereist: bool = Field(
        default=False,
        description="Of actie vereist is naar aanleiding van de reactie",
    )


class Rapport(BaseModel):
    """Een rapport van crew/agent naar management.

    Representeert een formele rapportage binnen de
    organisatiehiërarchie.

    Voorbeeld:
        ```python
        from crewai.communication import Rapport, RapportType, RapportPrioriteit

        rapport = Rapport(
            afzender_id=agent_id,
            ontvanger_ids=[manager_id],
            type=RapportType.STATUS,
            titel="Dagelijkse status update",
            samenvatting="Alle taken zijn op schema",
            prioriteit=RapportPrioriteit.INFO
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor dit rapport",
    )

    # Partijen
    afzender_id: UUID = Field(
        ...,
        description="ID van de afzender (crew of agent)",
    )
    afzender_type: Literal["agent", "crew"] = Field(
        default="agent",
        description="Type afzender",
    )
    ontvanger_ids: list[UUID] = Field(
        ...,
        description="IDs van de ontvangers (managers)",
    )

    # Inhoud
    type: RapportType = Field(
        ...,
        description="Type rapport",
    )
    titel: str = Field(
        ...,
        description="Titel van het rapport",
        min_length=1,
    )
    samenvatting: str = Field(
        ...,
        description="Korte samenvatting",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Gedetailleerde inhoud",
    )
    bijlagen: list[RapportBijlage] = Field(
        default_factory=list,
        description="Bijlagen bij het rapport",
    )

    # Prioriteit
    prioriteit: RapportPrioriteit = Field(
        default=RapportPrioriteit.NORMAAL,
        description="Prioriteit van het rapport",
    )

    # Gerelateerd
    opdracht_id: UUID | None = Field(
        default=None,
        description="ID van gerelateerde opdracht",
    )
    gerelateerde_rapport_ids: list[UUID] = Field(
        default_factory=list,
        description="IDs van gerelateerde rapporten",
    )

    # Tracking
    aangemaakt_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van aanmaken",
    )
    gelezen_door: list[UUID] = Field(
        default_factory=list,
        description="IDs van ontvangers die het rapport gelezen hebben",
    )
    gelezen_op: dict[str, datetime] = Field(
        default_factory=dict,
        description="Tijdstippen waarop ontvangers het rapport gelezen hebben",
    )

    # Reacties
    reacties: list[RapportReactie] = Field(
        default_factory=list,
        description="Reacties op het rapport",
    )

    # Metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Tags voor categorisatie",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata",
    )

    def markeer_gelezen(self, lezer_id: UUID) -> None:
        """Markeer het rapport als gelezen door een ontvanger.

        Args:
            lezer_id: ID van de lezer.
        """
        if lezer_id not in self.gelezen_door:
            self.gelezen_door.append(lezer_id)
            self.gelezen_op[str(lezer_id)] = datetime.now(timezone.utc)

    def voeg_reactie_toe(
        self,
        auteur_id: UUID,
        inhoud: str,
        actie_vereist: bool = False,
    ) -> RapportReactie:
        """Voeg een reactie toe aan het rapport.

        Args:
            auteur_id: ID van de auteur.
            inhoud: Inhoud van de reactie.
            actie_vereist: Of actie vereist is.

        Returns:
            De nieuwe reactie.
        """
        reactie = RapportReactie(
            auteur_id=auteur_id,
            inhoud=inhoud,
            actie_vereist=actie_vereist,
        )
        self.reacties.append(reactie)
        return reactie

    def voeg_bijlage_toe(
        self,
        naam: str,
        type: str,
        inhoud: Any,
        beschrijving: str | None = None,
    ) -> RapportBijlage:
        """Voeg een bijlage toe aan het rapport.

        Args:
            naam: Naam van de bijlage.
            type: Type bijlage.
            inhoud: Inhoud van de bijlage.
            beschrijving: Optionele beschrijving.

        Returns:
            De nieuwe bijlage.
        """
        bijlage = RapportBijlage(
            naam=naam,
            type=type,
            inhoud=inhoud,
            beschrijving=beschrijving,
        )
        self.bijlagen.append(bijlage)
        return bijlage

    def is_gelezen_door_allen(self) -> bool:
        """Check of alle ontvangers het rapport gelezen hebben.

        Returns:
            True als alle ontvangers gelezen hebben.
        """
        return all(
            ontvanger_id in self.gelezen_door for ontvanger_id in self.ontvanger_ids
        )

    def vereist_actie(self) -> bool:
        """Check of het rapport actie vereist.

        Returns:
            True als actie vereist is.
        """
        # Urgente rapporten vereisen altijd actie
        if self.prioriteit == RapportPrioriteit.URGENT:
            return True

        # Probleem en escalatie rapporten vereisen actie
        if self.type in [RapportType.PROBLEEM, RapportType.ESCALATIE]:
            return True

        # Check reacties
        return any(r.actie_vereist for r in self.reacties)


class RapportManager(BaseModel):
    """Beheert rapportages in de organisatie.

    Deze klasse coördineert het creëren, versturen en volgen
    van rapporten tussen crews/agents en management.

    Voorbeeld:
        ```python
        from crewai.communication import RapportManager, RapportType

        manager = RapportManager()

        # Stuur rapport
        rapport = manager.stuur_rapport(
            van_id=agent_id,
            naar_ids=[manager_id],
            type=RapportType.STATUS,
            titel="Dagelijkse update",
            samenvatting="Alles gaat goed"
        )

        # Krijg ongelezen rapporten
        ongelezen = manager.krijg_ongelezen_rapporten(manager_id)
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    rapporten: dict[UUID, Rapport] = Field(
        default_factory=dict,
        description="Alle rapporten geïndexeerd op ID",
    )

    # Organisatie referentie
    organisatie: "OrganisatieHierarchie | None" = Field(
        default=None,
        description="Referentie naar organisatie voor permissies",
    )

    # Callbacks
    on_nieuw_rapport: Callable[[Rapport], None] | None = Field(
        default=None,
        description="Callback bij nieuw rapport",
    )
    on_reactie: Callable[[Rapport, RapportReactie], None] | None = Field(
        default=None,
        description="Callback bij nieuwe reactie",
    )

    def stuur_rapport(
        self,
        van_id: UUID,
        naar_ids: list[UUID],
        type: RapportType,
        titel: str,
        samenvatting: str,
        details: dict[str, Any] | None = None,
        prioriteit: RapportPrioriteit = RapportPrioriteit.NORMAAL,
        opdracht_id: UUID | None = None,
        afzender_type: Literal["agent", "crew"] = "agent",
        tags: list[str] | None = None,
    ) -> Rapport:
        """Maak en verstuur een nieuw rapport.

        Args:
            van_id: ID van de afzender.
            naar_ids: IDs van de ontvangers.
            type: Type rapport.
            titel: Titel van het rapport.
            samenvatting: Samenvatting.
            details: Optionele details.
            prioriteit: Prioriteit.
            opdracht_id: Optioneel gerelateerde opdracht ID.
            afzender_type: Type afzender.
            tags: Optionele tags.

        Returns:
            Het nieuwe rapport.
        """
        # Controleer permissies indien organisatie beschikbaar
        if self.organisatie is not None:
            for naar_id in naar_ids:
                if not self.organisatie.mag_rapporteren_aan(van_id, naar_id):
                    raise PermissionError(
                        f"Agent {van_id} mag niet rapporteren aan {naar_id}"
                    )

        rapport = Rapport(
            afzender_id=van_id,
            afzender_type=afzender_type,
            ontvanger_ids=naar_ids,
            type=type,
            titel=titel,
            samenvatting=samenvatting,
            details=details or {},
            prioriteit=prioriteit,
            opdracht_id=opdracht_id,
            tags=tags or [],
        )

        self.rapporten[rapport.id] = rapport

        # Trigger callback
        if self.on_nieuw_rapport:
            self.on_nieuw_rapport(rapport)

        return rapport

    def krijg_rapport(self, rapport_id: UUID) -> Rapport | None:
        """Krijg een rapport op ID.

        Args:
            rapport_id: ID van het rapport.

        Returns:
            Het rapport of None.
        """
        return self.rapporten.get(rapport_id)

    def markeer_gelezen(self, rapport_id: UUID, lezer_id: UUID) -> Rapport | None:
        """Markeer een rapport als gelezen.

        Args:
            rapport_id: ID van het rapport.
            lezer_id: ID van de lezer.

        Returns:
            Het bijgewerkte rapport of None.
        """
        rapport = self.rapporten.get(rapport_id)
        if rapport is None:
            return None

        rapport.markeer_gelezen(lezer_id)
        return rapport

    def reageer_op_rapport(
        self,
        rapport_id: UUID,
        auteur_id: UUID,
        inhoud: str,
        actie_vereist: bool = False,
    ) -> RapportReactie | None:
        """Voeg een reactie toe aan een rapport.

        Args:
            rapport_id: ID van het rapport.
            auteur_id: ID van de auteur.
            inhoud: Inhoud van de reactie.
            actie_vereist: Of actie vereist is.

        Returns:
            De nieuwe reactie of None.
        """
        rapport = self.rapporten.get(rapport_id)
        if rapport is None:
            return None

        reactie = rapport.voeg_reactie_toe(auteur_id, inhoud, actie_vereist)

        # Trigger callback
        if self.on_reactie:
            self.on_reactie(rapport, reactie)

        return reactie

    # === QUERY METHODES ===

    def krijg_ongelezen_rapporten(self, voor_id: UUID) -> list[Rapport]:
        """Krijg alle ongelezen rapporten voor een ontvanger.

        Args:
            voor_id: ID van de ontvanger.

        Returns:
            Lijst van ongelezen rapporten.
        """
        return [
            r
            for r in self.rapporten.values()
            if voor_id in r.ontvanger_ids and voor_id not in r.gelezen_door
        ]

    def krijg_rapporten_voor(
        self,
        ontvanger_id: UUID,
        type: RapportType | None = None,
        alleen_ongelezen: bool = False,
    ) -> list[Rapport]:
        """Krijg alle rapporten voor een ontvanger.

        Args:
            ontvanger_id: ID van de ontvanger.
            type: Optioneel filter op type.
            alleen_ongelezen: Alleen ongelezen rapporten.

        Returns:
            Lijst van rapporten.
        """
        rapporten = [
            r for r in self.rapporten.values() if ontvanger_id in r.ontvanger_ids
        ]

        if type is not None:
            rapporten = [r for r in rapporten if r.type == type]

        if alleen_ongelezen:
            rapporten = [r for r in rapporten if ontvanger_id not in r.gelezen_door]

        return sorted(rapporten, key=lambda r: r.aangemaakt_op, reverse=True)

    def krijg_rapporten_van(
        self,
        afzender_id: UUID,
        type: RapportType | None = None,
    ) -> list[Rapport]:
        """Krijg alle rapporten van een afzender.

        Args:
            afzender_id: ID van de afzender.
            type: Optioneel filter op type.

        Returns:
            Lijst van rapporten.
        """
        rapporten = [
            r for r in self.rapporten.values() if r.afzender_id == afzender_id
        ]

        if type is not None:
            rapporten = [r for r in rapporten if r.type == type]

        return sorted(rapporten, key=lambda r: r.aangemaakt_op, reverse=True)

    def krijg_rapporten_voor_opdracht(self, opdracht_id: UUID) -> list[Rapport]:
        """Krijg alle rapporten voor een opdracht.

        Args:
            opdracht_id: ID van de opdracht.

        Returns:
            Lijst van gerelateerde rapporten.
        """
        return [
            r for r in self.rapporten.values() if r.opdracht_id == opdracht_id
        ]

    def krijg_urgente_rapporten(self) -> list[Rapport]:
        """Krijg alle urgente rapporten.

        Returns:
            Lijst van urgente rapporten.
        """
        return [
            r
            for r in self.rapporten.values()
            if r.prioriteit == RapportPrioriteit.URGENT and not r.is_gelezen_door_allen()
        ]

    def krijg_rapporten_met_actie_vereist(self) -> list[Rapport]:
        """Krijg alle rapporten die actie vereisen.

        Returns:
            Lijst van rapporten die actie vereisen.
        """
        return [r for r in self.rapporten.values() if r.vereist_actie()]

    def zoek_rapporten(
        self,
        zoekterm: str,
        type: RapportType | None = None,
        tags: list[str] | None = None,
    ) -> list[Rapport]:
        """Zoek rapporten op basis van criteria.

        Args:
            zoekterm: Zoekterm voor titel en samenvatting.
            type: Optioneel filter op type.
            tags: Optionele filter op tags.

        Returns:
            Lijst van matchende rapporten.
        """
        resultaten = []
        zoekterm_lower = zoekterm.lower()

        for rapport in self.rapporten.values():
            # Check zoekterm
            if zoekterm_lower not in rapport.titel.lower():
                if zoekterm_lower not in rapport.samenvatting.lower():
                    continue

            # Check type
            if type is not None and rapport.type != type:
                continue

            # Check tags
            if tags is not None:
                if not any(tag in rapport.tags for tag in tags):
                    continue

            resultaten.append(rapport)

        return sorted(resultaten, key=lambda r: r.aangemaakt_op, reverse=True)

    def krijg_statistieken(self) -> dict[str, Any]:
        """Krijg statistieken over rapporten.

        Returns:
            Dictionary met statistieken.
        """
        alle = list(self.rapporten.values())

        return {
            "totaal": len(alle),
            "ongelezen": len(
                [r for r in alle if not r.is_gelezen_door_allen()]
            ),
            "actie_vereist": len([r for r in alle if r.vereist_actie()]),
            "per_type": {
                type.value: len([r for r in alle if r.type == type])
                for type in RapportType
            },
            "per_prioriteit": {
                prioriteit.value: len(
                    [r for r in alle if r.prioriteit == prioriteit]
                )
                for prioriteit in RapportPrioriteit
            },
        }


# Helper functies voor veelvoorkomende rapporten
def maak_status_rapport(
    van_id: UUID,
    naar_ids: list[UUID],
    status: str,
    voortgang: int = 0,
    problemen: list[str] | None = None,
) -> Rapport:
    """Maak een standaard statusrapport.

    Args:
        van_id: ID van de afzender.
        naar_ids: IDs van de ontvangers.
        status: Korte status beschrijving.
        voortgang: Voortgangspercentage.
        problemen: Optionele lijst van problemen.

    Returns:
        Het statusrapport.
    """
    return Rapport(
        afzender_id=van_id,
        ontvanger_ids=naar_ids,
        type=RapportType.STATUS,
        titel="Status Update",
        samenvatting=status,
        prioriteit=RapportPrioriteit.INFO,
        details={
            "voortgang_percentage": voortgang,
            "problemen": problemen or [],
        },
    )


def maak_probleem_rapport(
    van_id: UUID,
    naar_ids: list[UUID],
    probleem: str,
    impact: str,
    voorgestelde_oplossing: str | None = None,
) -> Rapport:
    """Maak een probleemrapport.

    Args:
        van_id: ID van de afzender.
        naar_ids: IDs van de ontvangers.
        probleem: Beschrijving van het probleem.
        impact: Impact van het probleem.
        voorgestelde_oplossing: Optionele voorgestelde oplossing.

    Returns:
        Het probleemrapport.
    """
    return Rapport(
        afzender_id=van_id,
        ontvanger_ids=naar_ids,
        type=RapportType.PROBLEEM,
        titel=f"Probleem: {probleem[:50]}",
        samenvatting=probleem,
        prioriteit=RapportPrioriteit.BELANGRIJK,
        details={
            "impact": impact,
            "voorgestelde_oplossing": voorgestelde_oplossing,
        },
    )


def maak_resultaat_rapport(
    van_id: UUID,
    naar_ids: list[UUID],
    opdracht_id: UUID,
    resultaat: str,
    succes: bool = True,
    metrics: dict[str, Any] | None = None,
) -> Rapport:
    """Maak een resultaatrapport.

    Args:
        van_id: ID van de afzender.
        naar_ids: IDs van de ontvangers.
        opdracht_id: ID van de gerelateerde opdracht.
        resultaat: Beschrijving van het resultaat.
        succes: Of de opdracht succesvol was.
        metrics: Optionele metrics.

    Returns:
        Het resultaatrapport.
    """
    return Rapport(
        afzender_id=van_id,
        ontvanger_ids=naar_ids,
        type=RapportType.RESULTAAT,
        titel=f"Resultaat: {'Succesvol' if succes else 'Niet succesvol'}",
        samenvatting=resultaat,
        prioriteit=RapportPrioriteit.NORMAAL,
        opdracht_id=opdracht_id,
        details={
            "succes": succes,
            "metrics": metrics or {},
        },
    )
