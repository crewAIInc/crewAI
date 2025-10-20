"""A2A (Agent-to-Agent) Protocol adapter for CrewAI.

This module provides integration with A2A protocol-compliant agents,
enabling CrewAI to orchestrate external agents like ServiceNow, Bedrock Agents,
Glean, and other A2A-compliant systems.

Example:
    ```python
    from crewai.agents.agent_adapters.a2a import A2AAgentAdapter

    # Create A2A agent
    servicenow_agent = A2AAgentAdapter(
        agent_card_url="https://servicenow.example.com/.well-known/agent-card.json",
        auth_token="your-token",
        role="ServiceNow Incident Manager",
        goal="Create and manage IT incidents",
        backstory="Expert at incident management",
    )

    # Use in crew
    crew = Crew(agents=[servicenow_agent], tasks=[task])
    ```
"""

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from crewai.agents.agent_adapters.a2a.a2a_adapter import A2AAgentAdapter
    from crewai.agents.agent_adapters.a2a.exceptions import (
        A2AAuthenticationError,
        A2AConfigurationError,
        A2AConnectionError,
        A2AError,
        A2AInputRequiredError,
        A2ATaskCanceledError,
        A2ATaskFailedError,
    )

__all__ = [
    "A2AAgentAdapter",
    "A2AAuthenticationError",
    "A2AConfigurationError",
    "A2AConnectionError",
    "A2AError",
    "A2AInputRequiredError",
    "A2ATaskCanceledError",
    "A2ATaskFailedError",
]


def __getattr__(name: str) -> type:
    """Lazy import attributes to avoid requiring a2a-sdk unless actually used."""
    if name == "A2AAgentAdapter":
        from crewai.agents.agent_adapters.a2a.a2a_adapter import A2AAgentAdapter

        return A2AAgentAdapter
    if name in (
        "A2AError",
        "A2AConfigurationError",
        "A2AConnectionError",
        "A2AInputRequiredError",
        "A2ATaskFailedError",
        "A2AAuthenticationError",
        "A2ATaskCanceledError",
    ):
        from crewai.agents.agent_adapters.a2a.exceptions import (
            A2AAuthenticationError,
            A2AConfigurationError,
            A2AConnectionError,
            A2AError,
            A2AInputRequiredError,
            A2ATaskCanceledError,
            A2ATaskFailedError,
        )

        exceptions = {
            "A2AError": A2AError,
            "A2AConfigurationError": A2AConfigurationError,
            "A2AConnectionError": A2AConnectionError,
            "A2AInputRequiredError": A2AInputRequiredError,
            "A2ATaskFailedError": A2ATaskFailedError,
            "A2AAuthenticationError": A2AAuthenticationError,
            "A2ATaskCanceledError": A2ATaskCanceledError,
        }
        return exceptions[name]
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
