"""Manager klasse voor crew-gerelateerde tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.tools.base_tool import BaseTool
from crewai.tools.crew_tools.delegate_to_crew_tool import DelegateToCrewTool
from crewai.utilities.i18n import I18N, get_i18n

if TYPE_CHECKING:
    from crewai.crew import Crew


class CrewTools:
    """Manager voor crew delegatie tools.

    Deze klasse creëert en beheert tools die agents in staat stellen
    om werk te delegeren naar andere crews.

    Voorbeeld:
        ```python
        from crewai.tools.crew_tools import CrewTools

        # Maak crew tools voor beschikbare sub-crews
        crew_tools = CrewTools(crews={
            "spot_trading": spot_crew,
            "futures_trading": futures_crew,
            "research": research_crew,
        })

        # Voeg toe aan agent tools
        agent_tools = crew_tools.tools()
        ```
    """

    def __init__(
        self,
        crews: dict[str, Any],
        i18n: I18N | None = None,
    ) -> None:
        """Initialiseer de CrewTools manager.

        Args:
            crews: Dictionary met crew namen als keys en Crew instances als values.
            i18n: Optionele I18N instance voor vertalingen.
        """
        self.crews = crews
        self.i18n = i18n if i18n is not None else get_i18n()

    def tools(self) -> list[BaseTool]:
        """Geef alle beschikbare crew delegatie tools.

        Returns:
            Lijst met BaseTool instances voor crew delegatie.
        """
        if not self.crews:
            return []

        return [
            DelegateToCrewTool(
                crews=self.crews,
                i18n=self.i18n,
            )
        ]
