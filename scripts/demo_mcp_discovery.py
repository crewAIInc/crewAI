#!/usr/bin/env python3
"""End-to-end demo: MCP tool discovery with BaseDiscoveryProvider and DynamicDiscoveryTool.

This script demonstrates:
  1. MCPDiscoveryProvider.search_semantic() — semantic search via the mcp-discovery API.
  2. DiscoveryEntry list — normalized results (name, description, source_uri, raw_metadata).
  3. DynamicDiscoveryTool — a CrewAI tool that runs discovery and optionally registers tools to an agent.

Requirements:
  - Network access to the mcp-discovery API (default: https://mcp-discovery-two.vercel.app).
  - Optional: set MCP_DISCOVERY_API_BASE_URL to use a different endpoint.

Run from repo root:
  python scripts/demo_mcp_discovery.py
  # or with uv:
  uv run python scripts/demo_mcp_discovery.py
"""

from __future__ import annotations

import asyncio
import sys


def _ensure_crewai():
    """Ensure we can import crewai; add lib/crewai to path if running from repo root."""
    try:
        import crewai  # noqa: F401
    except ImportError:
        # Allow running from repo root without installing: python scripts/demo_mcp_discovery.py
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        lib_crewai = repo_root / "lib" / "crewai"
        if lib_crewai.is_dir():
            sys.path.insert(0, str(lib_crewai.parent))
        else:
            raise


_ensure_crewai()

from crewai.tools import BaseDiscoveryProvider, DiscoveryEntry, DynamicDiscoveryTool
from crewai.tools.mcp_discovery_provider import MCPDiscoveryProvider


async def demo_search_semantic(provider: MCPDiscoveryProvider) -> None:
    """1. Run semantic search and print DiscoveryEntry list."""
    print("=" * 60)
    print("1. MCPDiscoveryProvider.search_semantic()")
    print("=" * 60)

    query = "send slack notifications"
    limit = 3

    try:
        entries = await provider.search_semantic(query, limit=limit)
    except Exception as e:
        print(f"  Error calling mcp-discovery API: {e}")
        print("  (Check network and that the API is reachable.)")
        return

    print(f"  Query: {query!r}  limit={limit}")
    print(f"  Found {len(entries)} recommendation(s):\n")

    for i, entry in enumerate(entries, 1):
        print(f"  [{i}] {entry.name}")
        print(f"      id: {entry.id}")
        print(f"      description: {entry.description[:80]}..." if len(entry.description or "") > 80 else f"      description: {entry.description or '(none)'}")
        print(f"      source_uri: {entry.source_uri}")
        print(f"      provider_id: {entry.provider_id}")
        if entry.raw_metadata.get("install_command"):
            print(f"      install_command: {entry.raw_metadata['install_command']}")
        print()

    if not entries:
        print("  (No entries returned; API may be empty or query may have no matches.)\n")


def demo_dynamic_discovery_tool(provider: BaseDiscoveryProvider) -> None:
    """2. Use DynamicDiscoveryTool: run discovery and print result (no agent registration)."""
    print("=" * 60)
    print("2. DynamicDiscoveryTool (discover only, no agent)")
    print("=" * 60)

    discovery_tool = DynamicDiscoveryTool(
        provider=provider,
        limit=2,
        agent=None,
        auto_register=False,
        name="dynamic_tool_discovery",
    )

    query = "database or postgres"
    print(f"  Calling tool with search_query={query!r} ...\n")

    try:
        result = discovery_tool.run(search_query=query, limit=2)
    except Exception as e:
        print(f"  Error: {e}")
        return

    print(f"  Result count: {result['count']}")
    print(f"  Registered to agent: {result['registered']}")
    print(f"  Summary: {result['summary']}\n")

    tools = result.get("tools") or []
    if tools:
        print("  Discovered tools (usable as CrewAI BaseTool):")
        for t in tools:
            print(f"    - {t.name}: {t.description[:60]}..." if len(t.description or "") > 60 else f"    - {t.name}: {t.description or '(no description)'}")
    else:
        print("  (No tools resolved; entries may lack a valid MCP server URL for resolve_to_tool.)")
    print()


def demo_dynamic_discovery_with_agent(provider: BaseDiscoveryProvider) -> None:
    """3. Wire DynamicDiscoveryTool to an agent (tool can register discovered tools to agent.tools)."""
    print("=" * 60)
    print("3. DynamicDiscoveryTool + Agent (auto_register)")
    print("=" * 60)

    try:
        from crewai import Agent
    except ImportError:
        print("  Skipped: Agent not available (optional for this demo).\n")
        return

    # Agent starts with only the discovery tool
    agent = Agent(
        role="Tool discovery assistant",
        goal="Find and register useful MCP tools for the user.",
        backstory="You help users discover tools by natural language.",
        tools=[],  # We add DynamicDiscoveryTool below
        verbose=False,
    )

    discovery_tool = DynamicDiscoveryTool(
        provider=provider,
        limit=2,
        agent=agent,
        auto_register=True,
    )

    agent.tools = [discovery_tool]

    print("  Agent has one tool: dynamic_tool_discovery.")
    print("  When the agent uses it with a query, discovered tools are appended to agent.tools.")
    print("  Example: result = discovery_tool.run(search_query='slack')")
    print()

    try:
        result = discovery_tool.run(search_query="slack or notifications", limit=2)
        print(f"  After run: result['count']={result['count']}, result['registered']={result['registered']}")
        print(f"  agent.tools now has {len(agent.tools)} item(s):")
        for i, t in enumerate(agent.tools):
            name = getattr(t, "name", str(t))[:50]
            print(f"    [{i+1}] {name}")
    except Exception as e:
        print(f"  Error: {e}")

    print()


def main() -> None:
    print("\nMCP Tool Discovery — E2E Demo\n")

    provider = MCPDiscoveryProvider()
    print(f"Using MCPDiscoveryProvider (provider_id={provider.provider_id})")
    print()

    asyncio.run(demo_search_semantic(provider))

    demo_dynamic_discovery_tool(provider)

    demo_dynamic_discovery_with_agent(provider)

    print("Done.\n")


if __name__ == "__main__":
    main()
