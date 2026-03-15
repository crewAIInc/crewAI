"""Example: Using Joy trust verification with CrewAI.

This example shows how to verify agent trustworthiness before delegation
using the Joy trust network.

Install: pip install crewai[joy]

Environment:
    JOY_API_URL: Joy API endpoint (default: https://joy-connect.fly.dev)
    JOY_AGENT_ID: Your agent's Joy identity (optional)
"""

from crewai import Agent, Crew, Task
from crewai.trust import JoyVerifier, TrustVerificationError


def main():
    """Run a crew with Joy trust verification."""

    # Initialize the trust verifier
    # Requires min trust score of 0.5 (has at least some vouches)
    verifier = JoyVerifier(min_trust_score=0.5)

    # Example: Verify an agent before adding to crew
    agent_id = "ag_229e507d7d87f35cc2bc17ea"  # GitHub MCP agent

    result = verifier.verify_agent(agent_id)
    print(f"Agent: {agent_id}")
    print(f"  Trust Score: {result.trust_score}")
    print(f"  Vouch Count: {result.vouch_count}")
    print(f"  Verified: {result.verified}")
    print(f"  Trusted: {result.is_trusted}")
    print(f"  Capabilities: {result.capabilities}")

    # Example: Discover trusted agents for a capability
    print("\nDiscovering trusted agents with 'github' capability:")
    trusted_agents = verifier.discover_trusted_agents(capability="github", limit=5)
    for agent in trusted_agents:
        print(f"  - {agent.agent_id}: score={agent.trust_score}, vouches={agent.vouch_count}")

    # Example: Use verification before delegation
    try:
        verifier.verify_before_delegation(
            agent_id=agent_id,
            required_capabilities=["github"]
        )
        print(f"\nAgent {agent_id} passed verification - safe to delegate!")
    except TrustVerificationError as e:
        print(f"\nAgent failed verification: {e}")

    # Clean up
    verifier.close()


def crew_with_trust_verification():
    """Example of creating a crew with trust-verified agents."""

    verifier = JoyVerifier(min_trust_score=1.0)

    # Only create agents from trusted Joy identities
    trusted_ids = [
        "ag_229e507d7d87f35cc2bc17ea",  # GitHub MCP
        "ag_eae8bed274d50e29b05593e0",  # AWS MCP
    ]

    verified_agents = []
    for agent_id in trusted_ids:
        result = verifier.verify_agent(agent_id)
        if result.is_trusted:
            verified_agents.append({
                "joy_id": agent_id,
                "trust_score": result.trust_score,
                "capabilities": result.capabilities,
            })
            print(f"Verified: {agent_id} (score: {result.trust_score})")
        else:
            print(f"Rejected: {agent_id} (score: {result.trust_score})")

    print(f"\nVerified {len(verified_agents)} agents for crew")

    # Now create your CrewAI agents based on verified identities
    # ...

    verifier.close()


if __name__ == "__main__":
    main()
    print("\n" + "=" * 50 + "\n")
    crew_with_trust_verification()
