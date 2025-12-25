"""STAFF Crew (Group Executive Board) for QRI Trading Organization."""

from crewai import Crew, Process

from krakenagents.agents.staff import (
    create_ceo_agent,
    create_group_cio_agent,
    create_group_cro_agent,
    create_group_coo_agent,
    create_group_cfo_agent,
    create_compliance_agent,
    create_security_agent,
    create_prime_agent,
    create_data_agent,
    create_people_agent,
)
from krakenagents.tasks.staff_tasks import get_staff_tasks
from krakenagents.config import get_settings, get_chat_llm


def create_staff_crew() -> Crew:
    """Create the STAFF crew (Group Executive Board).

    Hierarchical crew with CEO as manager.
    Handles governance, risk oversight, and strategic decisions.

    Returns:
        Configured Crew instance.
    """
    settings = get_settings()

    # Create all STAFF agents
    ceo = create_ceo_agent()
    group_cio = create_group_cio_agent()
    group_cro = create_group_cro_agent()
    group_coo = create_group_coo_agent()
    group_cfo = create_group_cfo_agent()
    compliance = create_compliance_agent()
    security = create_security_agent()
    prime = create_prime_agent()
    data = create_data_agent()
    people = create_people_agent()

    # Note: manager_agent (ceo) should NOT be in agents list for hierarchical process
    agents = [
        group_cio,
        group_cro,
        group_coo,
        group_cfo,
        compliance,
        security,
        prime,
        data,
        people,
    ]

    # Create tasks for all agents
    agents_dict = {
        "ceo": ceo,
        "group_cio": group_cio,
        "group_cro": group_cro,
        "group_coo": group_coo,
        "group_cfo": group_cfo,
        "compliance": compliance,
        "security": security,
        "prime": prime,
        "data": data,
        "people": people,
    }
    tasks = get_staff_tasks(agents_dict)

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.hierarchical,
        manager_agent=ceo,
        chat_llm=get_chat_llm(),
        verbose=settings.crew_verbose,
        memory=settings.crew_memory,
        max_rpm=settings.crew_max_rpm,
        stream=True,
    )
