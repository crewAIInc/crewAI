"""
SwiftAPI Integration for CrewAI

Provides cryptographic attestation for CrewAI tool invocations and crew execution.
Every action is verified against SwiftAPI policies before execution.
No attestation, no execution.

Usage:
    # Wrap a crew with attestation
    from crewai import Crew, Agent, Task
    from crewai.swiftapi_integration import SwiftAPICrew

    crew = Crew(agents=[...], tasks=[...])
    swiftapi_crew = SwiftAPICrew(
        crew=crew,
        swiftapi_key="swiftapi_live_..."  # or set SWIFTAPI_KEY env var
    )
    result = swiftapi_crew.kickoff(inputs={...})

    # Or wrap individual tools
    from crewai.swiftapi_integration import SwiftAPIStructuredTool

    tool = SwiftAPIStructuredTool(
        name="my_tool",
        description="Does something",
        args_schema=MyArgs,
        func=my_func,
        swiftapi_key="swiftapi_live_..."
    )

Configuration:
    from crewai.swiftapi_integration import SwiftAPIConfig

    config = SwiftAPIConfig(
        api_key="swiftapi_live_...",    # required
        base_url="https://swiftapi.ai",  # default
        app_id="crewai",                 # default
        actor="crewai-agent",            # default
        timeout=10,                      # seconds
        fail_open=False,                 # NEVER set True in production
        verbose=True,                    # log attestation status
    )

Get a key: https://getswiftapi.com
"""

from .attestation import (
    AttestationError,
    AttestationProvider,
    AttestationResult,
    MockAttestationProvider,
    PolicyViolationError,
    SwiftAPIAttestationProvider,
)
from .config import SwiftAPIConfig
from .crew import SwiftAPICrew
from .tools import SwiftAPIStructuredTool, wrap_tools

__all__ = [
    # Config
    "SwiftAPIConfig",
    # Attestation
    "AttestationError",
    "AttestationProvider",
    "AttestationResult",
    "MockAttestationProvider",
    "PolicyViolationError",
    "SwiftAPIAttestationProvider",
    # Tools
    "SwiftAPIStructuredTool",
    "wrap_tools",
    # Crew
    "SwiftAPICrew",
]
