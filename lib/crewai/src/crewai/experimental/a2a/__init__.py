"""A2A (Agent-to-Agent) Protocol adapter for CrewAI.

This module provides integration with A2A protocol-compliant agents,
enabling CrewAI to orchestrate external agents like ServiceNow, Bedrock Agents,
Glean, and other A2A-compliant systems.

Example:
    ```python
    from crewai.experimental.a2a import A2AAgentAdapter

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

from crewai.experimental.a2a.a2a_adapter import A2AAgentAdapter
from crewai.experimental.a2a.auth import (
    APIKeyAuth,
    AuthScheme,
    BearerTokenAuth,
    HTTPBasicAuth,
    HTTPDigestAuth,
    OAuth2AuthorizationCode,
    OAuth2ClientCredentials,
    create_auth_from_agent_card,
)
from crewai.experimental.a2a.exceptions import (
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
    "APIKeyAuth",
    # Authentication
    "AuthScheme",
    "BearerTokenAuth",
    "HTTPBasicAuth",
    "HTTPDigestAuth",
    "OAuth2AuthorizationCode",
    "OAuth2ClientCredentials",
    "create_auth_from_agent_card",
]
