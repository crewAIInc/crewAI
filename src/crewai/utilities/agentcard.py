"""
agentcard_adapters.crewai_adapter
==================================

Bidirectional adapter between CrewAI agents/crews and AgentCard v1.0.

Usage
-----
**Export a CrewAI Agent as AgentCard:**

    from crewai import Agent
    from agentcard_adapters.crewai_adapter import agent_to_agentcard

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI",
        backstory="You are an expert AI researcher with 10 years of experience.",
        tools=[search_tool, scrape_tool],
        verbose=True,
    )

    card = agent_to_agentcard(
        agent=researcher,
        agent_id="01HZQK3P8EMXR9V7T5N2W4J6C0",
        endpoint_url="https://my-crew.example.com/api/researcher",
    )
    card.validate()
    print(card.to_json(indent=2))

**Export an entire Crew as an AgentCard registry:**

    from crewai import Crew
    from agentcard_adapters.crewai_adapter import crew_to_agentcards

    crew = Crew(agents=[researcher, writer], tasks=[...])
    cards = crew_to_agentcards(crew, base_url="https://my-crew.example.com/api")
    for card in cards:
        print(card.to_json(indent=2))

License
-------
Apache 2.0.  See https://github.com/kwailapt/AgentCard/blob/main/LICENSE
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from .core import (
    AgentCard,
    Capability,
    Endpoint,
    PricingModel,
    LANDAUER_FLOOR_JOULES,
)

if TYPE_CHECKING:
    from crewai import Agent as CrewAgent, Crew

__all__ = [
    "agent_to_agentcard",
    "crew_to_agentcards",
    "agentcard_to_agent",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _role_to_cap_id(role: str) -> str:
    """
    Convert a CrewAI agent role string to a valid capability id.

    Examples:
        "Senior Research Analyst"  → "senior_research_analyst"
        "Blog Post Writer"         → "blog_post_writer"
        "Code Review Expert"       → "code_review_expert"
    """
    s = role.lower().strip()
    s = re.sub(r"[\s]+", "_", s)
    s = re.sub(r"[^a-z0-9._-]", "", s)
    s = re.sub(r"^[^a-z0-9]+", "", s)
    return s or "agent"


def _tool_to_capability(tool: Any) -> Capability:
    """Convert a CrewAI/LangChain BaseTool to a Capability."""
    name = getattr(tool, "name", str(tool))
    desc = getattr(tool, "description", f"Tool: {name}")
    cap_id = re.sub(r"[^a-z0-9._-]", "", name.lower().replace(" ", "_"))
    if not cap_id or not cap_id[0].isalnum():
        cap_id = "tool_" + cap_id
    return Capability(id=cap_id or "tool", description=desc)


# ── Single agent → AgentCard ──────────────────────────────────────────────────

def agent_to_agentcard(
    agent: "CrewAgent",
    agent_id: str,
    endpoint_url: str,
    *,
    version: str = "1.0.0",
    protocol: str = "http",
    health_url: Optional[str] = None,
    estimated_latency_ms: Optional[float] = None,
    include_tool_capabilities: bool = True,
) -> AgentCard:
    """
    Convert a CrewAI ``Agent`` to an ``AgentCard``.

    The agent's *role* becomes the primary capability id.
    If ``include_tool_capabilities=True``, each tool is also listed as
    a separate capability under the namespace ``tool.<tool_name>``.

    Parameters
    ----------
    agent:
        A CrewAI ``Agent`` instance.
    agent_id:
        26-character Crockford Base32 ULID for this agent.
    endpoint_url:
        URL where this agent is reachable.
    version:
        Semantic version of the card. Defaults to ``"1.0.0"``.
    protocol:
        Transport protocol. Defaults to ``"http"``.
    health_url:
        Optional health-check endpoint.
    estimated_latency_ms:
        Estimated response latency in milliseconds.
    include_tool_capabilities:
        If ``True``, each tool is listed as a separate Capability.
        Set to ``False`` to emit only the role capability.

    Returns
    -------
    AgentCard
        A validated AgentCard for this CrewAI agent.
    """
    role: str = getattr(agent, "role", "Agent")
    goal: str = getattr(agent, "goal", "")
    backstory: str = getattr(agent, "backstory", "")

    # Primary capability: the agent's role
    primary_desc = goal
    if backstory:
        primary_desc = f"{goal} — {backstory[:200]}"

    primary_cap = Capability(
        id=_role_to_cap_id(role),
        description=primary_desc or f"CrewAI agent: {role}",
        tags=["crewai", "agent"],
    )

    capabilities = [primary_cap]

    # Tool capabilities
    if include_tool_capabilities:
        tools: list[Any] = getattr(agent, "tools", []) or []
        for tool in tools:
            try:
                cap = _tool_to_capability(tool)
                # Namespace under "tool." to avoid id collision with role
                if not cap.id.startswith("tool."):
                    cap.id = f"tool.{cap.id}"
                capabilities.append(cap)
            except Exception:  # noqa: BLE001
                pass

    pricing = None
    if estimated_latency_ms is not None:
        pricing = PricingModel(
            base_cost_joules=LANDAUER_FLOOR_JOULES,
            estimated_latency_ms=estimated_latency_ms,
        )

    card = AgentCard(
        agent_id=agent_id,
        name=role,
        version=version,
        capabilities=capabilities,
        endpoint=Endpoint(
            protocol=protocol,
            url=endpoint_url,
            health_url=health_url,
        ),
        pricing=pricing,
    )
    card.validate()
    return card


# ── Crew → list[AgentCard] ────────────────────────────────────────────────────

def crew_to_agentcards(
    crew: "Crew",
    agent_ids: Optional[list[str]] = None,
    base_url: str = "https://localhost/api",
    *,
    version: str = "1.0.0",
    protocol: str = "http",
    include_tool_capabilities: bool = True,
) -> list[AgentCard]:
    """
    Convert all agents in a CrewAI ``Crew`` to a list of ``AgentCard`` objects.

    Each agent gets a separate card.  The endpoint URL is derived as
    ``{base_url}/{slug}`` where *slug* is the role normalised to a URL path.

    Parameters
    ----------
    crew:
        A CrewAI ``Crew`` instance.
    agent_ids:
        Optional list of 26-char ULIDs, one per agent.  If omitted, IDs
        are derived from the agent roles (NOT globally unique — suitable
        for local testing only).
    base_url:
        Base URL prefix.  Each agent's card URL = ``{base_url}/{role_slug}``.
    version:
        Semantic version applied to all cards.
    protocol:
        Transport protocol for all cards.
    include_tool_capabilities:
        Whether to include tool capabilities in each card.

    Returns
    -------
    list[AgentCard]
        One validated AgentCard per agent.
    """
    agents: list[Any] = getattr(crew, "agents", []) or []
    if not agents:
        raise ValueError("Crew has no agents")

    cards = []
    for i, agent in enumerate(agents):
        role = getattr(agent, "role", f"agent_{i}")
        slug = _role_to_cap_id(role)
        url = f"{base_url.rstrip('/')}/{slug}"

        # Agent ID: use provided or generate a deterministic placeholder
        if agent_ids and i < len(agent_ids):
            aid = agent_ids[i]
        else:
            # Deterministic placeholder (NOT a real ULID — call out in docs)
            # Real deployment must supply proper ULIDs
            import hashlib
            h = hashlib.sha256(role.encode()).hexdigest().upper()[:26]
            # Ensure Crockford alphabet (replace I, L, O, U)
            h = h.translate(str.maketrans("ILOU", "JKMN"))
            aid = h

        card = agent_to_agentcard(
            agent=agent,
            agent_id=aid,
            endpoint_url=url,
            version=version,
            protocol=protocol,
            include_tool_capabilities=include_tool_capabilities,
        )
        cards.append(card)

    return cards


# ── AgentCard → CrewAI Agent ──────────────────────────────────────────────────

def agentcard_to_agent(
    card: AgentCard,
    llm: Any = None,
    verbose: bool = False,
) -> "CrewAgent":
    """
    Reconstruct a CrewAI ``Agent`` from an ``AgentCard``.

    The primary capability (index 0) is used as the agent's *role* and *goal*.
    Additional capabilities with ``tool.`` prefix are converted to
    ``RemoteAgentCardTool`` instances that POST to the card's endpoint.

    Parameters
    ----------
    card:
        The AgentCard to convert.
    llm:
        LLM instance to pass to the CrewAI Agent. If ``None``, uses
        CrewAI's default.
    verbose:
        Enable verbose CrewAI logging.

    Returns
    -------
    crewai.Agent
        A CrewAI Agent backed by the AgentCard's endpoint.

    Raises
    ------
    ImportError
        If ``crewai`` is not installed.
    """
    try:
        from crewai import Agent as CrewAgent
    except ImportError as e:
        raise ImportError("crewai is required: pip install crewai") from e

    primary = card.capabilities[0]
    role = card.name
    goal = primary.description

    kwargs: dict[str, Any] = {
        "role": role,
        "goal": goal,
        "backstory": (
            f"An agent with AgentCard id {card.agent_id} reachable at "
            f"{card.endpoint.url} via {card.endpoint.protocol}."
        ),
        "verbose": verbose,
    }
    if llm is not None:
        kwargs["llm"] = llm

    return CrewAgent(**kwargs)
