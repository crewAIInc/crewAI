"""Hiërarchische structuur en rapportagelijnen.

Dit module definieert de volledige organisatiestructuur inclusief
rapportagelijnen, hiërarchische relaties en validatiemethodes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from crewai.organization.department import Afdeling, AfdelingsManager
from crewai.organization.role import PermissieNiveau, Rol

if TYPE_CHECKING:
    pass


RapportageLijnType = Literal["direct", "functioneel", "dotted_line"]
"""Type rapportagelijn.

- direct: Vaste manager, dagelijkse aansturing
- functioneel: Voor specifieke taken of projecten
- dotted_line: Secundaire rapportage, adviserend
"""


class Rapportagelijn(BaseModel):
    """Definieert wie aan wie rapporteert.

    Een rapportagelijn verbindt een medewerker (agent of crew)
    met een manager in een specifieke afdeling.

    Voorbeeld:
        ```python
        from crewai.organization import Rapportagelijn

        lijn = Rapportagelijn(
            medewerker_id=agent_id,
            manager_id=manager_id,
            afdeling_id=trading_afd.id,
            type="direct"
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor deze rapportagelijn",
    )
    medewerker_id: UUID = Field(
        ...,
        description="ID van de agent of crew die rapporteert",
    )
    manager_id: UUID = Field(
        ...,
        description="ID van de manager agent/crew",
    )
    afdeling_id: UUID = Field(
        ...,
        description="ID van de afdeling waar deze relatie geldt",
    )

    type: RapportageLijnType = Field(
        default="direct",
        description="Type rapportagelijn",
    )

    # Tijdelijke rapportagelijnen (bijv. voor projecten)
    actief_sinds: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Starttijd van deze rapportagelijn",
    )
    actief_tot: datetime | None = Field(
        default=None,
        description="Eindtijd van deze rapportagelijn (None = permanent)",
    )

    # Metadata
    beschrijving: str | None = Field(
        default=None,
        description="Optionele beschrijving van de rapportagerelatie",
    )

    def is_actief(self) -> bool:
        """Controleer of deze rapportagelijn momenteel actief is.

        Returns:
            True als de lijn actief is.
        """
        nu = datetime.now(timezone.utc)
        if nu < self.actief_sinds:
            return False
        if self.actief_tot is not None and nu > self.actief_tot:
            return False
        return True


class OrganisatieHierarchie(BaseModel):
    """Beheert de volledige organisatiestructuur.

    Deze klasse is het centrale punt voor het beheren van de
    organisatiestructuur, inclusief afdelingen, rollen, en
    rapportagelijnen.

    Voorbeeld:
        ```python
        from crewai.organization import (
            OrganisatieHierarchie,
            Afdeling,
            Rol,
            PermissieNiveau
        )

        org = OrganisatieHierarchie(naam="Mijn Bedrijf")

        # Voeg afdelingen toe
        trading = Afdeling(naam="Trading")
        org.voeg_afdeling_toe(trading)

        # Definieer rollen
        manager_rol = Rol(
            naam="Trading Manager",
            niveau=PermissieNiveau.AFDELINGSHOOFD,
            kan_opdrachten_geven=True
        )
        org.wijs_rol_toe(agent_id, manager_rol)

        # Maak rapportagelijnen
        org.maak_rapportagelijn(
            medewerker_id=trader_id,
            manager_id=manager_id,
            afdeling_id=trading.id
        )
        ```
    """

    model_config = {"arbitrary_types_allowed": True}

    id: UUID = Field(
        default_factory=uuid4,
        description="Unieke identifier voor deze organisatie",
    )
    naam: str = Field(
        ...,
        description="Naam van de organisatie",
        min_length=1,
    )
    beschrijving: str | None = Field(
        default=None,
        description="Beschrijving van de organisatie",
    )

    # Structuur
    afdelingen_manager: AfdelingsManager = Field(
        default_factory=AfdelingsManager,
        description="Manager voor alle afdelingen",
    )
    rapportagelijnen: list[Rapportagelijn] = Field(
        default_factory=list,
        description="Alle rapportagelijnen in de organisatie",
    )
    agent_rollen: dict[UUID, Rol] = Field(
        default_factory=dict,
        description="Rollen toegewezen aan agents, geïndexeerd op agent ID",
    )
    crew_rollen: dict[UUID, Rol] = Field(
        default_factory=dict,
        description="Rollen toegewezen aan crews, geïndexeerd op crew ID",
    )

    # Configuratie
    standaard_isolatie: Literal["open", "afdeling", "strikt"] = Field(
        default="afdeling",
        description="Standaard isolatieniveau voor nieuwe afdelingen",
    )

    # Metadata
    aangemaakt_op: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Tijdstip van aanmaken",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra metadata",
    )

    # === AFDELING METHODES ===

    def voeg_afdeling_toe(self, afdeling: Afdeling) -> None:
        """Voeg een afdeling toe aan de organisatie.

        Args:
            afdeling: De afdeling om toe te voegen.
        """
        self.afdelingen_manager.voeg_toe(afdeling)

    def krijg_afdeling(self, afdeling_id: UUID) -> Afdeling | None:
        """Krijg een afdeling op ID.

        Args:
            afdeling_id: ID van de afdeling.

        Returns:
            De afdeling of None als niet gevonden.
        """
        return self.afdelingen_manager.krijg(afdeling_id)

    def krijg_afdeling_op_naam(self, naam: str) -> Afdeling | None:
        """Krijg een afdeling op naam.

        Args:
            naam: Naam van de afdeling.

        Returns:
            De afdeling of None als niet gevonden.
        """
        return self.afdelingen_manager.krijg_op_naam(naam)

    @property
    def afdelingen(self) -> dict[UUID, Afdeling]:
        """Krijg alle afdelingen."""
        return self.afdelingen_manager.afdelingen

    # === ROL METHODES ===

    def wijs_rol_toe(self, agent_id: UUID, rol: Rol) -> None:
        """Wijs een rol toe aan een agent.

        Args:
            agent_id: ID van de agent.
            rol: De rol om toe te wijzen.
        """
        self.agent_rollen[agent_id] = rol

    def krijg_rol(self, agent_id: UUID) -> Rol | None:
        """Krijg de rol van een agent.

        Args:
            agent_id: ID van de agent.

        Returns:
            De rol of None als geen rol toegewezen.
        """
        return self.agent_rollen.get(agent_id)

    def verwijder_rol(self, agent_id: UUID) -> bool:
        """Verwijder de rol van een agent.

        Args:
            agent_id: ID van de agent.

        Returns:
            True als verwijderd, False als geen rol gevonden.
        """
        if agent_id in self.agent_rollen:
            del self.agent_rollen[agent_id]
            return True
        return False

    # === CREW ROL METHODES ===

    def wijs_rol_toe_aan_crew(self, crew_id: UUID, rol: Rol) -> None:
        """Wijs een rol toe aan een crew.

        Dit bepaalt de permissies en bevoegdheden van de crew
        binnen de organisatie.

        Args:
            crew_id: ID van de crew.
            rol: De rol om toe te wijzen.

        Voorbeeld:
            ```python
            trading_rol = Rol(
                naam="Trading Afdeling",
                niveau=PermissieNiveau.AFDELINGSHOOFD,
                kan_opdrachten_geven=True,
                kan_rapporten_ontvangen=True
            )
            org.wijs_rol_toe_aan_crew(trading_crew.id, trading_rol)
            ```
        """
        self.crew_rollen[crew_id] = rol

    def krijg_crew_rol(self, crew_id: UUID) -> Rol | None:
        """Krijg de rol van een crew.

        Args:
            crew_id: ID van de crew.

        Returns:
            De rol of None als geen rol toegewezen.
        """
        return self.crew_rollen.get(crew_id)

    def verwijder_crew_rol(self, crew_id: UUID) -> bool:
        """Verwijder de rol van een crew.

        Args:
            crew_id: ID van de crew.

        Returns:
            True als verwijderd, False als geen rol gevonden.
        """
        if crew_id in self.crew_rollen:
            del self.crew_rollen[crew_id]
            return True
        return False

    def krijg_alle_crews_met_rol(self, niveau: PermissieNiveau) -> list[UUID]:
        """Krijg alle crews met een bepaald permissieniveau.

        Args:
            niveau: Het permissieniveau om te filteren.

        Returns:
            Lijst met crew IDs.
        """
        return [
            crew_id
            for crew_id, rol in self.crew_rollen.items()
            if rol.niveau == niveau
        ]

    def crew_mag_opdracht_geven(self, van_crew_id: UUID, naar_id: UUID) -> bool:
        """Controleer of een crew opdrachten mag geven aan een agent/crew.

        Args:
            van_crew_id: ID van de crew die opdracht wil geven.
            naar_id: ID van de ontvanger (agent of crew).

        Returns:
            True als toegestaan.
        """
        # Krijg rol van de verzendende crew
        van_rol = self.crew_rollen.get(van_crew_id)
        if van_rol is None:
            return False

        if not van_rol.kan_opdrachten_geven:
            return False

        # Check naar agent rol
        naar_agent_rol = self.agent_rollen.get(naar_id)
        if naar_agent_rol is not None:
            # Check of agent opdrachten ontvangt van dit niveau
            return van_rol.niveau in naar_agent_rol.ontvangt_opdrachten_van

        # Check naar crew rol
        naar_crew_rol = self.crew_rollen.get(naar_id)
        if naar_crew_rol is not None:
            # Hogere niveaus mogen opdrachten geven aan lagere
            niveau_volgorde = [
                PermissieNiveau.UITVOEREND,
                PermissieNiveau.TEAMLID,
                PermissieNiveau.TEAMLEIDER,
                PermissieNiveau.AFDELINGSHOOFD,
                PermissieNiveau.DIRECTIE,
                PermissieNiveau.BESTUUR,
            ]
            try:
                van_index = niveau_volgorde.index(van_rol.niveau)
                naar_index = niveau_volgorde.index(naar_crew_rol.niveau)
                return van_index > naar_index
            except ValueError:
                return False

        # Check rapportagelijnen als fallback
        return self.is_manager_van(van_crew_id, naar_id)

    def is_manager_van(self, manager_id: UUID, medewerker_id: UUID) -> bool:
        """Controleer of manager_id de manager is van medewerker_id.

        Args:
            manager_id: Potentiële manager ID.
            medewerker_id: Potentiële medewerker ID.

        Returns:
            True als manager_id de manager is.
        """
        for lijn in self.rapportagelijnen:
            if lijn.is_actief():
                if lijn.manager_id == manager_id and lijn.medewerker_id == medewerker_id:
                    return True
        return False

    # === RAPPORTAGELIJN METHODES ===

    def maak_rapportagelijn(
        self,
        medewerker_id: UUID,
        manager_id: UUID,
        afdeling_id: UUID,
        type: RapportageLijnType = "direct",
        **kwargs: Any,
    ) -> Rapportagelijn:
        """Maak een nieuwe rapportagelijn.

        Args:
            medewerker_id: ID van de medewerker.
            manager_id: ID van de manager.
            afdeling_id: ID van de afdeling.
            type: Type rapportagelijn.
            **kwargs: Extra argumenten voor Rapportagelijn.

        Returns:
            De nieuwe rapportagelijn.
        """
        lijn = Rapportagelijn(
            medewerker_id=medewerker_id,
            manager_id=manager_id,
            afdeling_id=afdeling_id,
            type=type,
            **kwargs,
        )
        self.rapportagelijnen.append(lijn)
        return lijn

    def verwijder_rapportagelijn(self, lijn_id: UUID) -> bool:
        """Verwijder een rapportagelijn.

        Args:
            lijn_id: ID van de rapportagelijn.

        Returns:
            True als verwijderd, False als niet gevonden.
        """
        for i, lijn in enumerate(self.rapportagelijnen):
            if lijn.id == lijn_id:
                self.rapportagelijnen.pop(i)
                return True
        return False

    # === HIËRARCHIE QUERIES ===

    def krijg_management_keten(self, agent_id: UUID) -> list[UUID]:
        """Krijg alle managers boven deze agent.

        Volgt de rapportagelijnen omhoog tot er geen manager meer is.

        Args:
            agent_id: ID van de agent.

        Returns:
            Lijst van manager IDs, van direct naar hoogste.
        """
        keten: list[UUID] = []
        huidige_id = agent_id
        bezocht: set[UUID] = set()

        while True:
            if huidige_id in bezocht:
                break  # Voorkom oneindige loops
            bezocht.add(huidige_id)

            # Zoek directe manager
            manager_id = None
            for lijn in self.rapportagelijnen:
                if lijn.medewerker_id == huidige_id and lijn.is_actief():
                    if lijn.type == "direct":
                        manager_id = lijn.manager_id
                        break

            if manager_id is None:
                break

            keten.append(manager_id)
            huidige_id = manager_id

        return keten

    def krijg_ondergeschikten(
        self, manager_id: UUID, alleen_direct: bool = False
    ) -> list[UUID]:
        """Krijg alle ondergeschikten van een manager.

        Args:
            manager_id: ID van de manager.
            alleen_direct: Als True, alleen directe ondergeschikten.

        Returns:
            Lijst van ondergeschikte IDs.
        """
        directe: list[UUID] = []

        for lijn in self.rapportagelijnen:
            if lijn.manager_id == manager_id and lijn.is_actief():
                if lijn.medewerker_id not in directe:
                    directe.append(lijn.medewerker_id)

        if alleen_direct:
            return directe

        # Recursief alle ondergeschikten verzamelen
        alle: list[UUID] = list(directe)
        for ondergeschikte_id in directe:
            sub_ondergeschikten = self.krijg_ondergeschikten(
                ondergeschikte_id, alleen_direct=False
            )
            for sub_id in sub_ondergeschikten:
                if sub_id not in alle:
                    alle.append(sub_id)

        return alle

    def krijg_managers(self, medewerker_id: UUID) -> list[UUID]:
        """Krijg alle managers van een medewerker.

        Inclusief dotted-line en functionele managers.

        Args:
            medewerker_id: ID van de medewerker.

        Returns:
            Lijst van manager IDs.
        """
        managers: list[UUID] = []

        for lijn in self.rapportagelijnen:
            if lijn.medewerker_id == medewerker_id and lijn.is_actief():
                if lijn.manager_id not in managers:
                    managers.append(lijn.manager_id)

        return managers

    def krijg_directe_manager(self, medewerker_id: UUID) -> UUID | None:
        """Krijg de directe manager van een medewerker.

        Args:
            medewerker_id: ID van de medewerker.

        Returns:
            ID van de directe manager of None.
        """
        for lijn in self.rapportagelijnen:
            if (
                lijn.medewerker_id == medewerker_id
                and lijn.type == "direct"
                and lijn.is_actief()
            ):
                return lijn.manager_id
        return None

    # === PERMISSIE CONTROLES ===

    def mag_opdracht_geven(self, van_id: UUID, naar_id: UUID) -> bool:
        """Controleer of van_id opdrachten mag geven aan naar_id.

        Controleert zowel rol-gebaseerde als hiërarchische permissies.

        Args:
            van_id: ID van de opdrachtgever.
            naar_id: ID van de opdrachtnemer.

        Returns:
            True als opdrachten geven is toegestaan.
        """
        # Krijg rollen
        van_rol = self.krijg_rol(van_id)
        naar_rol = self.krijg_rol(naar_id)

        # Zonder rol geen opdrachten geven
        if van_rol is None:
            return False

        # Check of van_rol opdrachten mag geven
        if not van_rol.kan_opdrachten_geven:
            return False

        # Check rol-gebaseerde permissie
        if naar_rol is not None:
            if van_rol.mag_opdracht_geven_aan(naar_rol):
                return True

        # Check hiërarchische relatie
        ondergeschikten = self.krijg_ondergeschikten(van_id)
        if naar_id in ondergeschikten:
            return True

        # Check of van_id manager is van naar_id
        managers = self.krijg_managers(naar_id)
        return van_id in managers

    def mag_rapporteren_aan(self, van_id: UUID, naar_id: UUID) -> bool:
        """Controleer of van_id mag rapporteren aan naar_id.

        Args:
            van_id: ID van de rapporteur.
            naar_id: ID van de ontvanger.

        Returns:
            True als rapporteren is toegestaan.
        """
        # Check of er een rapportagelijn bestaat
        for lijn in self.rapportagelijnen:
            if (
                lijn.medewerker_id == van_id
                and lijn.manager_id == naar_id
                and lijn.is_actief()
            ):
                return True

        # Check of naar_id in de management keten zit
        keten = self.krijg_management_keten(van_id)
        return naar_id in keten

    def mag_escaleren_naar(self, van_id: UUID, naar_id: UUID) -> bool:
        """Controleer of van_id mag escaleren naar naar_id.

        Args:
            van_id: ID van de escaleerder.
            naar_id: ID van het escalatiedoel.

        Returns:
            True als escalatie is toegestaan.
        """
        van_rol = self.krijg_rol(van_id)
        naar_rol = self.krijg_rol(naar_id)

        # Zonder rol geen escalatie
        if van_rol is None:
            return False

        # Check of van_rol mag escaleren
        if not van_rol.kan_escaleren:
            return False

        # Check naar hoger niveau
        if naar_rol is not None:
            if van_rol.mag_escaleren_naar(naar_rol.niveau):
                return True

        # Altijd kunnen escaleren naar management keten
        keten = self.krijg_management_keten(van_id)
        return naar_id in keten

    def mag_communiceren(self, van_id: UUID, naar_id: UUID) -> bool:
        """Controleer of twee entiteiten mogen communiceren.

        Houdt rekening met afdelingssisolatie.

        Args:
            van_id: ID van de afzender.
            naar_id: ID van de ontvanger.

        Returns:
            True als communicatie is toegestaan.
        """
        # Zoek afdelingen
        van_afd = self.afdelingen_manager.krijg_afdeling_voor_agent(van_id)
        naar_afd = self.afdelingen_manager.krijg_afdeling_voor_agent(naar_id)

        # Als geen afdeling, check crew afdelingen
        if van_afd is None:
            van_afd = self.afdelingen_manager.krijg_afdeling_voor_crew(van_id)
        if naar_afd is None:
            naar_afd = self.afdelingen_manager.krijg_afdeling_voor_crew(naar_id)

        # Als een van beiden geen afdeling heeft, sta toe
        if van_afd is None or naar_afd is None:
            return True

        # Check afdeling isolatie
        return van_afd.mag_communiceren_met(naar_afd)

    # === HELPER METHODES ===

    def krijg_permissie_niveau(self, agent_id: UUID) -> PermissieNiveau | None:
        """Krijg het permissieniveau van een agent.

        Args:
            agent_id: ID van de agent.

        Returns:
            Het permissieniveau of None als geen rol.
        """
        rol = self.krijg_rol(agent_id)
        if rol is None:
            return None
        return rol.niveau

    def valideer_structuur(self) -> list[str]:
        """Valideer de organisatiestructuur op problemen.

        Controleert op:
        - Circulaire rapportagelijnen
        - Medewerkers zonder manager
        - Onbekende afdelingen in rapportagelijnen

        Returns:
            Lijst van waarschuwingen/fouten.
        """
        problemen: list[str] = []

        # Check voor circulaire lijnen
        for lijn in self.rapportagelijnen:
            if lijn.medewerker_id == lijn.manager_id:
                problemen.append(
                    f"Circulaire rapportagelijn: {lijn.medewerker_id} rapporteert aan zichzelf"
                )

        # Check voor onbekende afdelingen
        for lijn in self.rapportagelijnen:
            if lijn.afdeling_id not in self.afdelingen:
                problemen.append(
                    f"Rapportagelijn {lijn.id} verwijst naar onbekende afdeling {lijn.afdeling_id}"
                )

        # Check voor medewerkers met meerdere directe managers
        medewerker_managers: dict[UUID, list[UUID]] = {}
        for lijn in self.rapportagelijnen:
            if lijn.type == "direct" and lijn.is_actief():
                if lijn.medewerker_id not in medewerker_managers:
                    medewerker_managers[lijn.medewerker_id] = []
                medewerker_managers[lijn.medewerker_id].append(lijn.manager_id)

        for medewerker_id, managers in medewerker_managers.items():
            if len(managers) > 1:
                problemen.append(
                    f"Medewerker {medewerker_id} heeft meerdere directe managers: {managers}"
                )

        return problemen

    def krijg_organigram(self) -> dict[str, Any]:
        """Genereer een organigram structuur.

        Returns:
            Dictionary met de organisatiestructuur voor visualisatie.
        """
        # Vind top-level managers (zonder eigen manager)
        alle_medewerkers: set[UUID] = set()
        alle_managers: set[UUID] = set()

        for lijn in self.rapportagelijnen:
            if lijn.is_actief():
                alle_medewerkers.add(lijn.medewerker_id)
                alle_managers.add(lijn.manager_id)

        top_managers = alle_managers - alle_medewerkers

        def bouw_boom(manager_id: UUID) -> dict[str, Any]:
            rol = self.krijg_rol(manager_id)
            ondergeschikten = self.krijg_ondergeschikten(manager_id, alleen_direct=True)

            return {
                "id": str(manager_id),
                "rol": rol.naam if rol else "Onbekend",
                "niveau": rol.niveau.value if rol else "onbekend",
                "ondergeschikten": [bouw_boom(o_id) for o_id in ondergeschikten],
            }

        return {
            "organisatie": self.naam,
            "afdelingen": [afd.naam for afd in self.afdelingen.values()],
            "hierarchie": [bouw_boom(m_id) for m_id in top_managers],
        }
