"""Pytest configuration for crewai-a2a tests.

Ensures Agent model is properly rebuilt with A2A types,
which can fail silently during circular import resolution.
"""

from crewai_a2a.config import A2AClientConfig, A2AConfig, A2AServerConfig


def pytest_configure() -> None:
    """Rebuild Agent/LiteAgent models after crewai_a2a is fully loaded."""
    from crewai.agent.core import Agent
    from crewai.lite_agent import LiteAgent

    ns = {
        "A2AConfig": A2AConfig,
        "A2AClientConfig": A2AClientConfig,
        "A2AServerConfig": A2AServerConfig,
    }
    Agent.model_rebuild(_types_namespace=ns)
    LiteAgent.model_rebuild(_types_namespace=ns)
