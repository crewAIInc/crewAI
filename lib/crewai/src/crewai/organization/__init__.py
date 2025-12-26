"""Organization module voor enterprise organisatiestructuren.

Dit module biedt een complete set klassen voor het modelleren van
organisatiestructuren zoals afdelingen, rollen, en hiërarchieën.

Voorbeeld:
    ```python
    from crewai.organization import (
        OrganisatieHierarchie,
        Afdeling,
        Rol,
        PermissieNiveau,
        Rapportagelijn,
    )

    # Maak organisatie
    org = OrganisatieHierarchie(naam="Mijn Bedrijf")

    # Maak afdeling
    trading = Afdeling(
        naam="Trading",
        beschrijving="Handelsactiviteiten",
        isolatie_niveau="afdeling"
    )
    org.voeg_afdeling_toe(trading)

    # Definieer rollen
    manager_rol = Rol(
        naam="Manager",
        niveau=PermissieNiveau.AFDELINGSHOOFD,
        kan_opdrachten_geven=True,
        kan_delegeren=True
    )

    medewerker_rol = Rol(
        naam="Medewerker",
        niveau=PermissieNiveau.TEAMLID,
        ontvangt_opdrachten_van=[PermissieNiveau.AFDELINGSHOOFD]
    )

    # Wijs rollen toe
    org.wijs_rol_toe(manager_id, manager_rol)
    org.wijs_rol_toe(medewerker_id, medewerker_rol)

    # Maak rapportagelijn
    org.maak_rapportagelijn(
        medewerker_id=medewerker_id,
        manager_id=manager_id,
        afdeling_id=trading.id,
        type="direct"
    )

    # Check permissies
    if org.mag_opdracht_geven(manager_id, medewerker_id):
        print("Manager mag opdrachten geven")
    ```
"""

from crewai.organization.department import (
    Afdeling,
    AfdelingsManager,
    AfdelingsResource,
    IsolatieNiveau,
)
from crewai.organization.hierarchy import (
    OrganisatieHierarchie,
    Rapportagelijn,
    RapportageLijnType,
)
from crewai.organization.role import (
    PermissieNiveau,
    Rol,
    maak_standaard_rollen,
)

__all__ = [
    # Role module
    "PermissieNiveau",
    "Rol",
    "maak_standaard_rollen",
    # Department module
    "Afdeling",
    "AfdelingsManager",
    "AfdelingsResource",
    "IsolatieNiveau",
    # Hierarchy module
    "OrganisatieHierarchie",
    "Rapportagelijn",
    "RapportageLijnType",
]
