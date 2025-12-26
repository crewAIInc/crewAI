"""Communication module voor opdrachten en rapportages.

Dit module biedt een complete set klassen voor communicatie
tussen crews, agents en management in de organisatiehiërarchie.

Voorbeeld:
    ```python
    from crewai.communication import (
        OpdrachtManager,
        Opdracht,
        OpdrachtPrioriteit,
        OpdrachtStatus,
        RapportManager,
        Rapport,
        RapportType,
        RapportPrioriteit,
    )

    # Setup opdracht manager
    opdracht_mgr = OpdrachtManager()

    # Geef opdracht
    opdracht = opdracht_mgr.geef_opdracht(
        van_id=manager_id,
        naar_id=medewerker_id,
        titel="Maak analyse",
        beschrijving="Analyseer de data van Q4",
        prioriteit=OpdrachtPrioriteit.HOOG
    )

    # Volg voortgang
    opdracht_mgr.update_voortgang(opdracht.id, 50, "Halverwege")

    # Setup rapport manager
    rapport_mgr = RapportManager()

    # Stuur rapport
    rapport = rapport_mgr.stuur_rapport(
        van_id=medewerker_id,
        naar_ids=[manager_id],
        type=RapportType.STATUS,
        titel="Dagelijkse update",
        samenvatting="Alle taken zijn op schema"
    )
    ```
"""

from crewai.communication.directives import (
    # Enums
    Opdracht,
    OpdrachtManager,
    OpdrachtPrioriteit,
    OpdrachtStatus,
    OpdrachtVoortgang,
)
from crewai.communication.reports import (
    # Klassen
    Rapport,
    RapportBijlage,
    RapportManager,
    # Enums
    RapportPrioriteit,
    RapportReactie,
    RapportType,
    # Helper functies
    maak_probleem_rapport,
    maak_resultaat_rapport,
    maak_status_rapport,
)
from crewai.communication.crew_channel import (
    # Klassen
    CrewBericht,
    CrewCommunicatieKanaal,
    # Enums
    BerichtPrioriteit,
    BerichtType,
)

__all__ = [
    # Directives
    "Opdracht",
    "OpdrachtManager",
    "OpdrachtPrioriteit",
    "OpdrachtStatus",
    "OpdrachtVoortgang",
    # Reports
    "Rapport",
    "RapportBijlage",
    "RapportManager",
    "RapportPrioriteit",
    "RapportReactie",
    "RapportType",
    # Report helpers
    "maak_probleem_rapport",
    "maak_resultaat_rapport",
    "maak_status_rapport",
    # Crew Channel
    "CrewBericht",
    "CrewCommunicatieKanaal",
    "BerichtPrioriteit",
    "BerichtType",
]
