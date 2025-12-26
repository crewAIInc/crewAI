"""Rollen en permissies voor agents in organisaties.

Dit module definieert het rollen- en permissiesysteem waarmee agents
in een hiërarchische organisatiestructuur kunnen opereren.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class PermissieNiveau(str, Enum):
    """Toegangsniveaus in de organisatie.

    Deze enum definieert de verschillende niveaus van autoriteit
    binnen een organisatie, van uitvoerend tot bestuursniveau.
    """

    UITVOEREND = "uitvoerend"
    """Alleen uitvoerende taken, geen delegatiebevoegdheid."""

    TEAMLID = "teamlid"
    """Kan binnen eigen team beperkt delegeren."""

    TEAMLEIDER = "teamleider"
    """Kan team aansturen en taken toewijzen."""

    AFDELINGSHOOFD = "afdelingshoofd"
    """Kan gehele afdeling aansturen en budget beheren."""

    DIRECTIE = "directie"
    """Kan alle afdelingen aansturen en strategische beslissingen nemen."""

    BESTUUR = "bestuur"
    """Volledige controle over de gehele organisatie."""

    @classmethod
    def is_hoger_dan(cls, niveau1: "PermissieNiveau", niveau2: "PermissieNiveau") -> bool:
        """Controleer of niveau1 hoger is dan niveau2.

        Args:
            niveau1: Eerste permissieniveau om te vergelijken.
            niveau2: Tweede permissieniveau om te vergelijken.

        Returns:
            True als niveau1 hoger is dan niveau2.
        """
        volgorde = [
            cls.UITVOEREND,
            cls.TEAMLID,
            cls.TEAMLEIDER,
            cls.AFDELINGSHOOFD,
            cls.DIRECTIE,
            cls.BESTUUR,
        ]
        return volgorde.index(niveau1) > volgorde.index(niveau2)

    @classmethod
    def is_gelijk_of_hoger_dan(
        cls, niveau1: "PermissieNiveau", niveau2: "PermissieNiveau"
    ) -> bool:
        """Controleer of niveau1 gelijk aan of hoger is dan niveau2.

        Args:
            niveau1: Eerste permissieniveau om te vergelijken.
            niveau2: Tweede permissieniveau om te vergelijken.

        Returns:
            True als niveau1 gelijk aan of hoger is dan niveau2.
        """
        return niveau1 == niveau2 or cls.is_hoger_dan(niveau1, niveau2)


class Rol(BaseModel):
    """Gedetailleerde rol met permissies voor een agent.

    Een rol definieert wat een agent mag doen binnen de organisatie,
    inclusief welke tools toegankelijk zijn, of ze mogen delegeren,
    en van wie ze opdrachten mogen ontvangen.

    Voorbeeld:
        ```python
        from crewai.organization import Rol, PermissieNiveau

        teamleider = Rol(
            naam="Teamleider Trading",
            niveau=PermissieNiveau.TEAMLEIDER,
            kan_delegeren=True,
            kan_opdrachten_geven=True,
            ontvangt_opdrachten_van=[
                PermissieNiveau.AFDELINGSHOOFD,
                PermissieNiveau.DIRECTIE
            ]
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor deze rol",
    )
    naam: str = Field(
        ...,
        description="Naam van de rol",
        min_length=1,
    )
    beschrijving: str | None = Field(
        default=None,
        description="Optionele beschrijving van de rol en verantwoordelijkheden",
    )
    niveau: PermissieNiveau = Field(
        default=PermissieNiveau.TEAMLID,
        description="Hiërarchisch niveau van deze rol",
    )

    # Tool permissies
    toegestane_tools: list[str] = Field(
        default_factory=list,
        description="Lijst van tools die deze rol mag gebruiken (leeg = alle tools)",
    )
    geblokkeerde_tools: list[str] = Field(
        default_factory=list,
        description="Lijst van tools die deze rol niet mag gebruiken",
    )

    # Organisatie permissies
    kan_delegeren: bool = Field(
        default=False,
        description="Of deze rol taken mag delegeren naar anderen",
    )
    kan_escaleren: bool = Field(
        default=True,
        description="Of deze rol problemen mag escaleren naar management",
    )
    kan_goedkeuren: bool = Field(
        default=False,
        description="Of deze rol verzoeken en voorstellen mag goedkeuren",
    )
    kan_opdrachten_geven: bool = Field(
        default=False,
        description="Of deze rol opdrachten mag geven aan anderen",
    )
    kan_rapporten_ontvangen: bool = Field(
        default=False,
        description="Of deze rol rapporten mag ontvangen van ondergeschikten",
    )

    # Relaties
    ontvangt_opdrachten_van: list[PermissieNiveau] = Field(
        default_factory=list,
        description="Niveaus waarvan deze rol opdrachten mag ontvangen",
    )
    escaleert_naar: list[PermissieNiveau] = Field(
        default_factory=list,
        description="Niveaus waarnaar deze rol mag escaleren",
    )

    # Resource limieten
    max_budget: float | None = Field(
        default=None,
        description="Maximaal budget dat deze rol mag aanwenden (None = onbeperkt)",
    )
    max_gelijktijdige_taken: int = Field(
        default=5,
        description="Maximum aantal taken dat tegelijk uitgevoerd mag worden",
    )

    # Metadata
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata voor de rol",
    )

    def mag_tool_gebruiken(self, tool_naam: str) -> bool:
        """Controleer of deze rol een specifieke tool mag gebruiken.

        Args:
            tool_naam: Naam van de tool om te controleren.

        Returns:
            True als de tool gebruikt mag worden.
        """
        # Geblokkeerde tools hebben voorrang
        if tool_naam in self.geblokkeerde_tools:
            return False

        # Als toegestane_tools leeg is, zijn alle tools toegestaan
        if not self.toegestane_tools:
            return True

        return tool_naam in self.toegestane_tools

    def mag_opdracht_geven_aan(self, andere_rol: "Rol") -> bool:
        """Controleer of deze rol opdrachten mag geven aan een andere rol.

        Args:
            andere_rol: De rol om te controleren.

        Returns:
            True als opdrachten gegeven mogen worden.
        """
        if not self.kan_opdrachten_geven:
            return False

        # Check of ons niveau in de ontvangt_opdrachten_van van de andere rol staat
        return self.niveau in andere_rol.ontvangt_opdrachten_van

    def mag_escaleren_naar(self, niveau: PermissieNiveau) -> bool:
        """Controleer of deze rol mag escaleren naar een bepaald niveau.

        Args:
            niveau: Het niveau om naar te escaleren.

        Returns:
            True als escalatie toegestaan is.
        """
        if not self.kan_escaleren:
            return False

        # Als escaleert_naar leeg is, mag naar elk hoger niveau
        if not self.escaleert_naar:
            return PermissieNiveau.is_hoger_dan(niveau, self.niveau)

        return niveau in self.escaleert_naar


# Voorgedefinieerde standaard rollen
def maak_standaard_rollen() -> dict[str, Rol]:
    """Maak een set standaard rollen voor een typische organisatie.

    Returns:
        Dictionary met standaard rollen geïndexeerd op naam.
    """
    return {
        "bestuurder": Rol(
            naam="Bestuurder",
            beschrijving="Hoogste beslissingsbevoegdheid in de organisatie",
            niveau=PermissieNiveau.BESTUUR,
            kan_delegeren=True,
            kan_goedkeuren=True,
            kan_opdrachten_geven=True,
            kan_rapporten_ontvangen=True,
            ontvangt_opdrachten_van=[],  # Niemand geeft bestuur opdrachten
            escaleert_naar=[],  # Bestuur escaleert niet
        ),
        "directeur": Rol(
            naam="Directeur",
            beschrijving="Leidt de dagelijkse operaties van de organisatie",
            niveau=PermissieNiveau.DIRECTIE,
            kan_delegeren=True,
            kan_goedkeuren=True,
            kan_opdrachten_geven=True,
            kan_rapporten_ontvangen=True,
            ontvangt_opdrachten_van=[PermissieNiveau.BESTUUR],
            escaleert_naar=[PermissieNiveau.BESTUUR],
        ),
        "afdelingshoofd": Rol(
            naam="Afdelingshoofd",
            beschrijving="Leidt een specifieke afdeling",
            niveau=PermissieNiveau.AFDELINGSHOOFD,
            kan_delegeren=True,
            kan_goedkeuren=True,
            kan_opdrachten_geven=True,
            kan_rapporten_ontvangen=True,
            ontvangt_opdrachten_van=[
                PermissieNiveau.DIRECTIE,
                PermissieNiveau.BESTUUR,
            ],
            escaleert_naar=[PermissieNiveau.DIRECTIE],
        ),
        "teamleider": Rol(
            naam="Teamleider",
            beschrijving="Leidt een team binnen een afdeling",
            niveau=PermissieNiveau.TEAMLEIDER,
            kan_delegeren=True,
            kan_goedkeuren=False,
            kan_opdrachten_geven=True,
            kan_rapporten_ontvangen=True,
            ontvangt_opdrachten_van=[
                PermissieNiveau.AFDELINGSHOOFD,
                PermissieNiveau.DIRECTIE,
            ],
            escaleert_naar=[PermissieNiveau.AFDELINGSHOOFD],
        ),
        "teamlid": Rol(
            naam="Teamlid",
            beschrijving="Standaard teamlid met uitvoerende taken",
            niveau=PermissieNiveau.TEAMLID,
            kan_delegeren=False,
            kan_goedkeuren=False,
            kan_opdrachten_geven=False,
            kan_rapporten_ontvangen=False,
            ontvangt_opdrachten_van=[
                PermissieNiveau.TEAMLEIDER,
                PermissieNiveau.AFDELINGSHOOFD,
            ],
            escaleert_naar=[PermissieNiveau.TEAMLEIDER],
        ),
        "uitvoerend": Rol(
            naam="Uitvoerend Medewerker",
            beschrijving="Alleen uitvoerende taken, geen coördinatie",
            niveau=PermissieNiveau.UITVOEREND,
            kan_delegeren=False,
            kan_goedkeuren=False,
            kan_opdrachten_geven=False,
            kan_rapporten_ontvangen=False,
            ontvangt_opdrachten_van=[
                PermissieNiveau.TEAMLID,
                PermissieNiveau.TEAMLEIDER,
            ],
            escaleert_naar=[PermissieNiveau.TEAMLID],
        ),
    }
