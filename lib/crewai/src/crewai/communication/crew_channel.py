"""Real-time communicatiekanaal tussen crews.

Dit module biedt functionaliteit voor directe berichten uitwisseling
tussen crews tijdens runtime, buiten het formele opdracht/rapport systeem.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    pass


class BerichtType(str, Enum):
    """Type van een crew bericht."""

    INFORMATIE = "informatie"
    VRAAG = "vraag"
    ANTWOORD = "antwoord"
    VERZOEK = "verzoek"
    BEVESTIGING = "bevestiging"
    WAARSCHUWING = "waarschuwing"
    STATUS = "status"


class BerichtPrioriteit(str, Enum):
    """Prioriteit van een bericht."""

    LAAG = "laag"
    NORMAAL = "normaal"
    HOOG = "hoog"
    URGENT = "urgent"


class CrewBericht(BaseModel):
    """Enkel bericht tussen crews.

    Dit representeert een direct bericht van de ene crew naar de andere,
    bedoeld voor snelle communicatie tijdens runtime.

    Voorbeeld:
        ```python
        bericht = CrewBericht(
            van_crew_id=trading_crew.id,
            naar_crew_id=risk_crew.id,
            inhoud="Grote positie geopend in EUR/USD",
            type=BerichtType.INFORMATIE
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor dit bericht",
    )
    van_crew_id: UUID = Field(
        ...,
        description="ID van de verzendende crew",
    )
    naar_crew_id: UUID = Field(
        ...,
        description="ID van de ontvangende crew",
    )
    inhoud: str = Field(
        ...,
        description="Inhoud van het bericht",
        min_length=1,
    )

    type: BerichtType = Field(
        default=BerichtType.INFORMATIE,
        description="Type bericht",
    )
    prioriteit: BerichtPrioriteit = Field(
        default=BerichtPrioriteit.NORMAAL,
        description="Prioriteit van het bericht",
    )

    # Context
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra context data",
    )
    antwoord_op: UUID | None = Field(
        default=None,
        description="ID van bericht waarop dit een antwoord is",
    )

    # Tracking
    tijdstip: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van verzenden",
    )
    gelezen: bool = Field(
        default=False,
        description="Of het bericht is gelezen",
    )
    gelezen_op: datetime | None = Field(
        default=None,
        description="Tijdstip van lezen",
    )

    def markeer_gelezen(self) -> None:
        """Markeer dit bericht als gelezen."""
        self.gelezen = True
        self.gelezen_op = datetime.now(timezone.utc)

    def is_urgent(self) -> bool:
        """Check of dit bericht urgent is."""
        return self.prioriteit in (BerichtPrioriteit.HOOG, BerichtPrioriteit.URGENT)


class CrewCommunicatieKanaal(BaseModel):
    """Real-time communicatiekanaal tussen crews.

    Dit kanaal faciliteert directe berichten uitwisseling tussen crews
    tijdens runtime. Het is bedoeld voor snelle, informele communicatie
    buiten het formele opdracht/rapport systeem.

    Voorbeeld:
        ```python
        from crewai.communication import CrewCommunicatieKanaal

        # Maak kanaal
        kanaal = CrewCommunicatieKanaal(naam="Trading-Risk Kanaal")

        # Voeg deelnemers toe
        kanaal.voeg_deelnemer_toe(trading_crew.id)
        kanaal.voeg_deelnemer_toe(risk_crew.id)

        # Stuur bericht
        kanaal.stuur_bericht(
            van_crew_id=trading_crew.id,
            naar_crew_id=risk_crew.id,
            inhoud="Vraag: Wat is de huidige risico limiet?",
            type=BerichtType.VRAAG
        )

        # Ontvang berichten
        berichten = kanaal.ontvang_berichten(risk_crew.id)
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor dit kanaal",
    )
    naam: str = Field(
        default="Crew Kanaal",
        description="Naam van het communicatiekanaal",
    )

    # Deelnemers
    deelnemers: list[UUID] = Field(
        default_factory=list,
        description="Crew IDs die deelnemen aan dit kanaal",
    )

    # Berichten
    berichten: list[CrewBericht] = Field(
        default_factory=list,
        description="Alle berichten in dit kanaal",
    )

    # Configuratie
    max_berichten: int = Field(
        default=1000,
        description="Maximum aantal berichten om te bewaren",
    )
    archiveer_oude_berichten: bool = Field(
        default=True,
        description="Of oude berichten gearchiveerd moeten worden",
    )

    # Callbacks
    _on_nieuw_bericht: list[Callable[[CrewBericht], None]] = []

    # Metadata
    aangemaakt_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van aanmaken",
    )

    def voeg_deelnemer_toe(self, crew_id: UUID) -> bool:
        """Voeg een crew toe als deelnemer.

        Args:
            crew_id: ID van de crew om toe te voegen.

        Returns:
            True als toegevoegd, False als al deelnemer.
        """
        if crew_id not in self.deelnemers:
            self.deelnemers.append(crew_id)
            return True
        return False

    def verwijder_deelnemer(self, crew_id: UUID) -> bool:
        """Verwijder een crew als deelnemer.

        Args:
            crew_id: ID van de crew om te verwijderen.

        Returns:
            True als verwijderd, False als niet gevonden.
        """
        if crew_id in self.deelnemers:
            self.deelnemers.remove(crew_id)
            return True
        return False

    def is_deelnemer(self, crew_id: UUID) -> bool:
        """Check of een crew deelnemer is.

        Args:
            crew_id: ID van de crew om te checken.

        Returns:
            True als deelnemer.
        """
        return crew_id in self.deelnemers

    def stuur_bericht(
        self,
        van_crew_id: UUID,
        naar_crew_id: UUID,
        inhoud: str,
        type: BerichtType = BerichtType.INFORMATIE,
        prioriteit: BerichtPrioriteit = BerichtPrioriteit.NORMAAL,
        context: dict[str, Any] | None = None,
        antwoord_op: UUID | None = None,
    ) -> CrewBericht:
        """Stuur een bericht naar een andere crew.

        Args:
            van_crew_id: ID van de verzendende crew.
            naar_crew_id: ID van de ontvangende crew.
            inhoud: Inhoud van het bericht.
            type: Type bericht.
            prioriteit: Prioriteit van het bericht.
            context: Extra context data.
            antwoord_op: ID van bericht waarop dit een antwoord is.

        Returns:
            Het aangemaakte bericht.

        Raises:
            PermissionError: Als verzender geen deelnemer is.
            ValueError: Als ontvanger geen deelnemer is.
        """
        # Valideer deelnemers
        if not self.is_deelnemer(van_crew_id):
            raise PermissionError(
                f"Crew {van_crew_id} is geen deelnemer van dit kanaal"
            )

        if not self.is_deelnemer(naar_crew_id):
            raise ValueError(
                f"Ontvanger {naar_crew_id} is geen deelnemer van dit kanaal"
            )

        # Maak bericht
        bericht = CrewBericht(
            van_crew_id=van_crew_id,
            naar_crew_id=naar_crew_id,
            inhoud=inhoud,
            type=type,
            prioriteit=prioriteit,
            context=context or {},
            antwoord_op=antwoord_op,
        )

        # Voeg toe aan berichten
        self.berichten.append(bericht)

        # Beperk aantal berichten indien nodig
        if len(self.berichten) > self.max_berichten:
            self._archiveer_berichten()

        # Trigger callbacks
        for callback in self._on_nieuw_bericht:
            try:
                callback(bericht)
            except Exception:
                pass

        return bericht

    def ontvang_berichten(
        self,
        crew_id: UUID,
        alleen_ongelezen: bool = False,
        markeer_gelezen: bool = True,
        type_filter: BerichtType | None = None,
    ) -> list[CrewBericht]:
        """Ontvang berichten voor een crew.

        Args:
            crew_id: ID van de ontvangende crew.
            alleen_ongelezen: Alleen ongelezen berichten.
            markeer_gelezen: Markeer opgehaalde berichten als gelezen.
            type_filter: Filter op bericht type.

        Returns:
            Lijst met berichten voor deze crew.
        """
        resultaat = []

        for bericht in self.berichten:
            if bericht.naar_crew_id != crew_id:
                continue

            if alleen_ongelezen and bericht.gelezen:
                continue

            if type_filter is not None and bericht.type != type_filter:
                continue

            resultaat.append(bericht)

            if markeer_gelezen and not bericht.gelezen:
                bericht.markeer_gelezen()

        return resultaat

    def krijg_gesprek(
        self,
        crew_a: UUID,
        crew_b: UUID,
        limiet: int = 50,
    ) -> list[CrewBericht]:
        """Krijg alle berichten tussen twee crews.

        Args:
            crew_a: ID van eerste crew.
            crew_b: ID van tweede crew.
            limiet: Maximum aantal berichten.

        Returns:
            Lijst met berichten tussen de crews, gesorteerd op tijd.
        """
        resultaat = []

        for bericht in self.berichten:
            is_match = (
                (bericht.van_crew_id == crew_a and bericht.naar_crew_id == crew_b)
                or (bericht.van_crew_id == crew_b and bericht.naar_crew_id == crew_a)
            )
            if is_match:
                resultaat.append(bericht)

        # Sorteer op tijd en beperk
        resultaat.sort(key=lambda b: b.tijdstip)
        return resultaat[-limiet:]

    def krijg_onbeantwoorde_vragen(self, crew_id: UUID) -> list[CrewBericht]:
        """Krijg alle onbeantwoorde vragen voor een crew.

        Args:
            crew_id: ID van de crew.

        Returns:
            Lijst met onbeantwoorde vragen.
        """
        # Verzamel vraag IDs
        vraag_ids = set()
        for bericht in self.berichten:
            if bericht.naar_crew_id == crew_id and bericht.type == BerichtType.VRAAG:
                vraag_ids.add(bericht.id)

        # Verwijder beantwoorde vragen
        for bericht in self.berichten:
            if bericht.antwoord_op in vraag_ids:
                vraag_ids.discard(bericht.antwoord_op)

        # Krijg onbeantwoorde vragen
        return [
            b for b in self.berichten
            if b.id in vraag_ids
        ]

    def beantwoord(
        self,
        vraag_id: UUID,
        van_crew_id: UUID,
        antwoord: str,
    ) -> CrewBericht:
        """Beantwoord een vraag.

        Args:
            vraag_id: ID van de vraag om te beantwoorden.
            van_crew_id: ID van de beantwoordende crew.
            antwoord: Het antwoord.

        Returns:
            Het antwoord bericht.

        Raises:
            ValueError: Als de vraag niet gevonden wordt.
        """
        # Zoek de vraag
        vraag = None
        for bericht in self.berichten:
            if bericht.id == vraag_id:
                vraag = bericht
                break

        if vraag is None:
            raise ValueError(f"Vraag met ID {vraag_id} niet gevonden")

        # Stuur antwoord
        return self.stuur_bericht(
            van_crew_id=van_crew_id,
            naar_crew_id=vraag.van_crew_id,
            inhoud=antwoord,
            type=BerichtType.ANTWOORD,
            antwoord_op=vraag_id,
        )

    def broadcast(
        self,
        van_crew_id: UUID,
        inhoud: str,
        type: BerichtType = BerichtType.INFORMATIE,
        prioriteit: BerichtPrioriteit = BerichtPrioriteit.NORMAAL,
    ) -> list[CrewBericht]:
        """Stuur een bericht naar alle deelnemers.

        Args:
            van_crew_id: ID van de verzendende crew.
            inhoud: Inhoud van het bericht.
            type: Type bericht.
            prioriteit: Prioriteit van het bericht.

        Returns:
            Lijst met verstuurde berichten.
        """
        berichten = []

        for deelnemer_id in self.deelnemers:
            if deelnemer_id != van_crew_id:
                bericht = self.stuur_bericht(
                    van_crew_id=van_crew_id,
                    naar_crew_id=deelnemer_id,
                    inhoud=inhoud,
                    type=type,
                    prioriteit=prioriteit,
                )
                berichten.append(bericht)

        return berichten

    def registreer_callback(
        self,
        callback: Callable[[CrewBericht], None],
    ) -> None:
        """Registreer een callback voor nieuwe berichten.

        Args:
            callback: Functie om aan te roepen bij nieuwe berichten.
        """
        self._on_nieuw_bericht.append(callback)

    def _archiveer_berichten(self) -> None:
        """Archiveer oude berichten om ruimte te maken."""
        if not self.archiveer_oude_berichten:
            # Gewoon verwijderen
            self.berichten = self.berichten[-self.max_berichten:]
        else:
            # TODO: Implementeer archivering naar opslag
            self.berichten = self.berichten[-self.max_berichten:]

    def krijg_statistieken(self) -> dict[str, Any]:
        """Krijg statistieken over het kanaal.

        Returns:
            Dictionary met statistieken.
        """
        totaal = len(self.berichten)
        ongelezen = sum(1 for b in self.berichten if not b.gelezen)
        per_type = {}

        for bericht in self.berichten:
            type_naam = bericht.type.value
            per_type[type_naam] = per_type.get(type_naam, 0) + 1

        return {
            "kanaal_id": str(self.id),
            "kanaal_naam": self.naam,
            "aantal_deelnemers": len(self.deelnemers),
            "totaal_berichten": totaal,
            "ongelezen_berichten": ongelezen,
            "per_type": per_type,
        }
