"""Main entry point for QRI Trading Organization.

This module provides the main functions to run the trading organization
with different configurations.
"""

import argparse
import sys
from typing import Any


def run_organization(
    desk: str | None = None,
    minimal: bool = False,
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the trading organization.

    Args:
        desk: Specific desk to run ('staff', 'spot', 'futures', or None for all)
        minimal: Run minimal crew for testing
        inputs: Optional inputs for the crews

    Returns:
        Dictionary with execution results.
    """
    from krakenagents.crews import (
        create_staff_crew,
        create_spot_crew,
        create_futures_crew,
        create_trading_organization,
    )
    from krakenagents.crews.trading_org import create_minimal_crew

    if minimal:
        # Run minimal crew for testing
        crew = create_minimal_crew(desk=desk or "spot")
        result = crew.kickoff(inputs=inputs or {})
        return {"result": result, "type": "minimal", "desk": desk or "spot"}

    if desk:
        # Run specific desk
        if desk == "staff":
            crew = create_staff_crew()
        elif desk == "spot":
            crew = create_spot_crew()
        elif desk == "futures":
            crew = create_futures_crew()
        else:
            raise ValueError(f"Unknown desk: {desk}")

        result = crew.kickoff(inputs=inputs or {})
        return {"result": result, "type": "desk", "desk": desk}

    # Run full organization
    org = create_trading_organization()
    results = org["run_all"](inputs=inputs or {})
    return {"results": results, "type": "organization"}


def list_agents() -> None:
    """Print list of all agents in the organization."""
    from krakenagents.agents.staff import get_all_staff_agents
    from krakenagents.agents.spot import get_all_spot_agents
    from krakenagents.agents.futures import get_all_futures_agents

    print("\n=== QRI Trading Organization Agents ===\n")

    print("STAFF (Group Executive Board) - 10 agents:")
    print("-" * 50)
    for i, agent in enumerate(get_all_staff_agents()):
        print(f"  STAFF-{i:02d}: {agent.role.split(' — ')[0]}")

    print("\nSpot Desk - 32 agents:")
    print("-" * 50)
    for i, agent in enumerate(get_all_spot_agents(), start=1):
        print(f"  Agent {i:02d}: {agent.role.split(' — ')[0][:50]}")

    print("\nFutures Desk - 32 agents:")
    print("-" * 50)
    for i, agent in enumerate(get_all_futures_agents(), start=33):
        print(f"  Agent {i:02d}: {agent.role.split(' — ')[0][:50]}")

    print(f"\nTotal: 74 agents")


def check_config() -> None:
    """Check and print current configuration."""
    from krakenagents.config import get_settings

    settings = get_settings()

    print("\n=== QRI Trading Organization Configuration ===\n")

    print("Heavy LLM (Complex reasoning):")
    print(f"  Provider: {settings.llm_provider}")
    print(f"  Model: {settings.llm_model}")
    print(f"  Base URL: {settings.llm_base_url}")
    print(f"  Temperature: {settings.llm_temperature}")

    print("\nLight LLM (Simple tasks):")
    print(f"  Provider: {settings.llm_light_provider}")
    print(f"  Model: {settings.llm_light_model}")
    print(f"  Base URL: {settings.llm_light_base_url}")
    print(f"  Temperature: {settings.llm_light_temperature}")

    print("\nEmbeddings:")
    print(f"  Provider: {settings.embedder_provider}")
    print(f"  Model: {settings.embedder_model}")
    print(f"  Base URL: {settings.embedder_base_url}")

    print("\nKraken API:")
    print(f"  Spot API Key: {'***' + settings.kraken_api_key[-4:] if settings.kraken_api_key else 'NOT SET'}")
    print(f"  Futures API Key: {'***' + settings.kraken_futures_api_key[-4:] if settings.kraken_futures_api_key else 'NOT SET'}")

    print("\nRisk Settings:")
    print(f"  Mode: {settings.risk_mode}")
    print(f"  Max Leverage Spot: {settings.max_leverage_spot}x")
    print(f"  Max Leverage Futures: {settings.max_leverage_futures}x")

    print("\nCrew Settings:")
    print(f"  Verbose: {settings.crew_verbose}")
    print(f"  Memory: {settings.crew_memory}")
    print(f"  Max RPM: {settings.crew_max_rpm}")


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QRI Trading Organization - CrewAI Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m krakenagents --list          List all agents
  python -m krakenagents --config        Check configuration
  python -m krakenagents --desk spot     Run Spot desk only
  python -m krakenagents --minimal       Run minimal crew for testing
  python -m krakenagents                 Run full organization
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all agents in the organization",
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Check and print current configuration",
    )
    parser.add_argument(
        "--desk",
        choices=["staff", "spot", "futures"],
        help="Run specific desk only",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help="Run minimal crew for testing",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="JSON input for the crew",
    )

    args = parser.parse_args()

    if args.list:
        list_agents()
        return 0

    if args.config:
        check_config()
        return 0

    # Parse inputs if provided
    inputs = None
    if args.input:
        import json
        try:
            inputs = json.loads(args.input)
        except json.JSONDecodeError as e:
            print(f"Error parsing input JSON: {e}")
            return 1

    # Run the organization
    try:
        result = run_organization(
            desk=args.desk,
            minimal=args.minimal,
            inputs=inputs,
        )
        print("\n=== Execution Complete ===")
        print(f"Type: {result['type']}")
        if "desk" in result:
            print(f"Desk: {result['desk']}")
        print(f"Result: {result.get('result', result.get('results'))}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
