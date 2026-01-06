SwiftAPI Integration for CrewAI

Cryptographic attestation for CrewAI tool invocations and crew execution.
Every action requires authorization before execution. No attestation, no execution.

Files:
- config.py: SwiftAPIConfig dataclass
- attestation.py: HTTP client and attestation provider
- tools.py: SwiftAPIStructuredTool wrapping CrewStructuredTool
- crew.py: SwiftAPICrew wrapping Crew for multi-agent attestation
- demo.py: Standalone test against live SwiftAPI

Usage:

    from crewai import Crew, Agent, Task
    from crewai.swiftapi_integration import SwiftAPICrew

    crew = Crew(agents=[...], tasks=[...])
    swiftapi_crew = SwiftAPICrew(
        crew=crew,
        swiftapi_key="swiftapi_live_..."  # or set SWIFTAPI_KEY env var
    )
    result = swiftapi_crew.kickoff(inputs={...})

Or wrap individual tools:

    from crewai.swiftapi_integration import SwiftAPIStructuredTool

    tool = SwiftAPIStructuredTool(
        name="my_tool",
        description="Does something",
        args_schema=MyArgs,
        func=my_func,
        swiftapi_key="swiftapi_live_..."
    )

Configuration:

    SwiftAPIConfig(
        api_key="swiftapi_live_...",     # required
        base_url="https://swiftapi.ai",   # default
        app_id="crewai",                  # default
        actor="crewai-agent",             # default
        timeout=10,                        # seconds
        fail_open=False,                   # NEVER set True in production
        verbose=True,                      # log attestation status
    )

Test:

    export SWIFTAPI_KEY="swiftapi_live_..."
    python demo.py

Get a key: https://getswiftapi.com
