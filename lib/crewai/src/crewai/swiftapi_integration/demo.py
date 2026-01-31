#!/usr/bin/env python3
"""
SwiftAPI + CrewAI Integration Demo

Tests SwiftAPI attestation against live endpoint.
Requires: SWIFTAPI_KEY environment variable

Run standalone:
    export SWIFTAPI_KEY="swiftapi_live_..."
    python demo.py

Run as module (requires Python 3.10+ for CrewAI):
    python -m crewai.swiftapi_integration.demo
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone


async def test_direct_api(api_key: str) -> bool:
    """Test direct SwiftAPI connection."""
    print("\n" + "=" * 60)
    print("TEST: Direct SwiftAPI Connection")
    print("=" * 60)

    import httpx

    client = httpx.AsyncClient(
        base_url="https://swiftapi.ai",
        timeout=10,
        headers={
            "X-SwiftAPI-Authority": api_key,
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-SwiftAPI/1.0",
        },
    )

    request_id = f"crewai_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

    payload = {
        "action": {
            "type": "tool_invocation",
            "intent": "crewai demo testing attestation",
            "params": {"tool": "test_tool", "args": {"query": "test"}},
        },
        "context": {
            "app_id": "crewai",
            "actor": "demo_agent",
            "environment": "production",
            "request_id": request_id,
            "agent_name": "demo_agent",
            "crew_name": "demo_crew",
        },
    }

    try:
        response = await client.post("/verify", json=payload)
        if response.status_code == 200:
            data = response.json()
            jti = data.get("verification_id") or data.get("jti")
            print(f"[PASS] Attestation approved")
            print(f"       JTI: {jti}")
            print(f"       Reason: {data.get('reason', 'N/A')}")
            return True
        else:
            print(f"[FAIL] Status {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False
    finally:
        await client.aclose()


async def test_crew_kickoff(api_key: str) -> bool:
    """Test crew kickoff attestation."""
    print("\n" + "=" * 60)
    print("TEST: Crew Kickoff Attestation")
    print("=" * 60)

    import httpx

    client = httpx.AsyncClient(
        base_url="https://swiftapi.ai",
        timeout=10,
        headers={
            "X-SwiftAPI-Authority": api_key,
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-SwiftAPI/1.0",
        },
    )

    request_id = f"crewai_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

    payload = {
        "action": {
            "type": "crew_kickoff",
            "intent": "crew 'demo_crew' kickoff with 2 agents, 3 tasks",
            "params": {
                "crew_name": "demo_crew",
                "agent_count": 2,
                "task_count": 3,
                "process": "sequential",
            },
        },
        "context": {
            "app_id": "crewai",
            "actor": "crewai-agent",
            "environment": "production",
            "request_id": request_id,
            "crew_name": "demo_crew",
        },
    }

    try:
        response = await client.post("/verify", json=payload)
        if response.status_code == 200:
            data = response.json()
            jti = data.get("verification_id") or data.get("jti")
            print(f"[PASS] Crew kickoff approved")
            print(f"       JTI: {jti}")
            return True
        else:
            print(f"[FAIL] Status {response.status_code}: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False
    finally:
        await client.aclose()


async def test_multi_action(api_key: str) -> bool:
    """Test multiple actions in sequence."""
    print("\n" + "=" * 60)
    print("TEST: Multi-Action Sequence")
    print("=" * 60)

    import httpx

    client = httpx.AsyncClient(
        base_url="https://swiftapi.ai",
        timeout=10,
        headers={
            "X-SwiftAPI-Authority": api_key,
            "Content-Type": "application/json",
            "User-Agent": "CrewAI-SwiftAPI/1.0",
        },
    )

    actions = [
        ("crew_kickoff", "Starting multi-agent workflow"),
        ("tool_invocation", "Agent 1 using search tool"),
        ("tool_invocation", "Agent 2 using analysis tool"),
        ("agent_handoff", "Passing results to Agent 3"),
    ]

    jtis = []
    all_passed = True

    for action_type, intent in actions:
        request_id = f"crewai_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        payload = {
            "action": {
                "type": action_type,
                "intent": intent,
                "params": {"step": len(jtis) + 1},
            },
            "context": {
                "app_id": "crewai",
                "actor": "crewai-agent",
                "environment": "production",
                "request_id": request_id,
            },
        }

        try:
            response = await client.post("/verify", json=payload)
            if response.status_code == 200:
                data = response.json()
                jti = data.get("verification_id") or data.get("jti")
                jtis.append(jti)
                print(f"  Step {len(jtis)}: {action_type} -> {jti[:12]}...")
            else:
                print(f"  Step {len(jtis)+1}: FAILED ({response.status_code})")
                all_passed = False
        except Exception as e:
            print(f"  Step {len(jtis)+1}: ERROR ({e})")
            all_passed = False

    await client.aclose()

    if all_passed:
        print(f"\n[PASS] All {len(actions)} actions attested")
        print(f"       Audit trail: {len(jtis)} JTIs recorded")
    else:
        print(f"\n[FAIL] Some actions failed")

    return all_passed


async def main():
    """Run all tests."""
    print("SwiftAPI + CrewAI Integration Test")
    print("=" * 60)

    api_key = os.getenv("SWIFTAPI_KEY")
    if not api_key:
        print("ERROR: SWIFTAPI_KEY not set")
        print("Set SWIFTAPI_KEY environment variable to run tests.")
        return False

    print(f"API Key: {api_key[:20]}...{api_key[-8:]}")
    print(f"Endpoint: https://swiftapi.ai")

    results = []
    results.append(("Direct API", await test_direct_api(api_key)))
    results.append(("Crew Kickoff", await test_crew_kickoff(api_key)))
    results.append(("Multi-Action", await test_multi_action(api_key)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{len(results)} passed")
    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
