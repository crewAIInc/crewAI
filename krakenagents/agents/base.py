"""Base agent factory for creating agents with proper LLM configuration."""

from typing import Any

from crewai import Agent

from krakenagents.config.llm_config import get_heavy_llm, get_light_llm


def create_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: list | None = None,
    use_heavy_llm: bool = False,
    allow_delegation: bool = False,
    verbose: bool = True,
    **kwargs: Any,
) -> Agent:
    """Create an agent with the appropriate LLM configuration.

    Args:
        role: The agent's role/title.
        goal: What the agent is trying to achieve.
        backstory: The agent's background and expertise.
        tools: List of tools available to the agent.
        use_heavy_llm: Use heavy LLM for complex reasoning (default: light).
        allow_delegation: Allow agent to delegate to others.
        verbose: Enable verbose logging.
        **kwargs: Additional agent parameters.

    Returns:
        Configured Agent instance.
    """
    llm = get_heavy_llm() if use_heavy_llm else get_light_llm()

    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=llm,
        tools=tools or [],
        allow_delegation=allow_delegation,
        verbose=verbose,
        **kwargs,
    )


def create_heavy_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: list | None = None,
    allow_delegation: bool = True,
    **kwargs: Any,
) -> Agent:
    """Create an agent with heavy LLM for complex reasoning.

    Used for: Leadership, discretionary traders, research analysts.
    """
    return create_agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools,
        use_heavy_llm=True,
        allow_delegation=allow_delegation,
        **kwargs,
    )


def create_light_agent(
    role: str,
    goal: str,
    backstory: str,
    tools: list | None = None,
    allow_delegation: bool = False,
    **kwargs: Any,
) -> Agent:
    """Create an agent with light LLM for simple tasks.

    Used for: Systematic traders, operations, execution, monitoring.
    """
    return create_agent(
        role=role,
        goal=goal,
        backstory=backstory,
        tools=tools,
        use_heavy_llm=False,
        allow_delegation=allow_delegation,
        **kwargs,
    )
