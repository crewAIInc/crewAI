"""Organization tools voor enterprise organisatiestructuren.

Dit module bevat tools waarmee agents en crews kunnen communiceren
binnen de organisatiehiërarchie.

Voorbeeld:
    ```python
    from crewai.tools.organization_tools import (
        GeefOpdrachtTool,
        RapporteerTool,
        EscaleerTool,
        DelegeerNaarCrewTool,
        OrganizationTools,
    )

    # Maak tools voor een manager agent
    org_tools = OrganizationTools(
        organisatie=org,
        opdracht_manager=opdracht_mgr,
        rapport_manager=rapport_mgr,
        escalatie_manager=escalatie_mgr,
    )

    # Krijg alle tools
    tools = org_tools.tools()

    # Of maak een delegatie tool voor crew-to-crew communicatie
    delegeer_tool = DelegeerNaarCrewTool(
        crew_id=manager_crew.id,
        sub_crews={"trading": trading_crew}
    )
    ```
"""

from crewai.tools.organization_tools.delegate_to_crew_tool import DelegeerNaarCrewTool
from crewai.tools.organization_tools.escalate_tool import EscaleerTool
from crewai.tools.organization_tools.give_directive_tool import GeefOpdrachtTool
from crewai.tools.organization_tools.organization_tools import OrganizationTools
from crewai.tools.organization_tools.report_tool import RapporteerTool

__all__ = [
    "GeefOpdrachtTool",
    "RapporteerTool",
    "EscaleerTool",
    "DelegeerNaarCrewTool",
    "OrganizationTools",
]
