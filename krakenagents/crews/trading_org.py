"""Trading Organization master crew for QRI Trading Organization."""

from crewai import Crew, Process, Task

from krakenagents.agents.staff import create_ceo_agent
from krakenagents.crews.staff_crew import create_staff_crew
from krakenagents.crews.spot_crew import create_spot_crew
from krakenagents.crews.futures_crew import create_futures_crew
from krakenagents.config import get_settings


def create_trading_organization(
    include_staff: bool = True,
    include_spot: bool = True,
    include_futures: bool = True,
) -> dict:
    """Create the complete trading organization.

    This creates all crews and provides a unified interface
    for running the organization.

    Args:
        include_staff: Include the STAFF crew.
        include_spot: Include the Spot desk crew.
        include_futures: Include the Futures desk crew.

    Returns:
        Dictionary containing:
        - crews: Dict of crew name to Crew instance
        - ceo: The CEO agent for organization-wide coordination
        - run: Function to run a specific crew or all crews
    """
    settings = get_settings()
    crews = {}

    if include_staff:
        crews["staff"] = create_staff_crew()

    if include_spot:
        crews["spot"] = create_spot_crew()

    if include_futures:
        crews["futures"] = create_futures_crew()

    # Create CEO for organization-wide coordination
    ceo = create_ceo_agent()

    def run_crew(crew_name: str, inputs: dict | None = None) -> str:
        """Run a specific crew.

        Args:
            crew_name: Name of crew to run ('staff', 'spot', 'futures')
            inputs: Optional inputs for the crew

        Returns:
            Crew execution result.
        """
        if crew_name not in crews:
            raise ValueError(f"Unknown crew: {crew_name}. Available: {list(crews.keys())}")

        return crews[crew_name].kickoff(inputs=inputs or {})

    def run_all(inputs: dict | None = None) -> dict[str, str]:
        """Run all crews sequentially.

        Args:
            inputs: Optional inputs for all crews

        Returns:
            Dictionary mapping crew name to result.
        """
        results = {}
        for name, crew in crews.items():
            results[name] = crew.kickoff(inputs=inputs or {})
        return results

    return {
        "crews": crews,
        "ceo": ceo,
        "run": run_crew,
        "run_all": run_all,
        "settings": settings,
    }


def create_minimal_crew(desk: str = "spot") -> Crew:
    """Create a minimal crew for testing with just leadership agents.

    Args:
        desk: Which desk to create ('spot' or 'futures')

    Returns:
        Minimal Crew instance.
    """
    settings = get_settings()

    if desk == "spot":
        from krakenagents.agents.spot import (
            create_spot_cio_agent,
            create_spot_head_trading_agent,
            create_spot_cro_agent,
        )

        cio = create_spot_cio_agent()
        head = create_spot_head_trading_agent()
        cro = create_spot_cro_agent()

        agents = [cio, head, cro]

        tasks = [
            Task(
                description="Review current spot positions and exposure.",
                expected_output="Position and exposure summary.",
                agent=cio,
            ),
            Task(
                description="Check trading quality and any rule violations.",
                expected_output="Trading quality report.",
                agent=head,
            ),
            Task(
                description="Verify all positions are within risk limits.",
                expected_output="Risk compliance report.",
                agent=cro,
            ),
        ]

    else:  # futures
        from krakenagents.agents.futures import (
            create_futures_cio_agent,
            create_futures_head_trading_agent,
            create_futures_cro_agent,
        )

        cio = create_futures_cio_agent()
        head = create_futures_head_trading_agent()
        cro = create_futures_cro_agent()

        agents = [cio, head, cro]

        tasks = [
            Task(
                description="Review current futures positions and leverage.",
                expected_output="Position and leverage summary.",
                agent=cio,
            ),
            Task(
                description="Check funding rates and trading quality.",
                expected_output="Funding and trading report.",
                agent=head,
            ),
            Task(
                description="Verify margin utilization and liquidation distance.",
                expected_output="Margin risk report.",
                agent=cro,
            ),
        ]

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=settings.crew_verbose,
        memory=settings.crew_memory,
    )
