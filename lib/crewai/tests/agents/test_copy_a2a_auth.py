"""Agent.copy must preserve concrete A2A auth scheme instances."""

from crewai import Agent
from crewai.a2a.auth.client_schemes import BearerTokenAuth
from crewai.a2a.config import A2AClientConfig


def test_copy_preserves_a2a_auth():
    agent = Agent(
        role="r",
        goal="g",
        backstory="b",
        llm="gpt-4o-mini",
        a2a=A2AClientConfig(
            endpoint="http://example.com/.well-known/agent-card.json",
            auth=BearerTokenAuth(token="t"),
        ),
    )

    copied = agent.copy()

    assert isinstance(copied.a2a.auth, BearerTokenAuth)
    assert copied.a2a.auth.token == "t"
