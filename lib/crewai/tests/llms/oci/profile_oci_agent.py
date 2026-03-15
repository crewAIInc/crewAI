from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crewai import Agent, LLM
from crewai.tools import tool

os.environ.setdefault("OTEL_SDK_DISABLED", "true")


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the sum."""
    return a + b


@dataclass
class ScenarioResult:
    name: str
    elapsed_seconds: float
    responses: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile the OCI tool-calling agent flow used by the OCI integration tests."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OCI_TEST_TOOL_MODEL", "openai.gpt-5.2-chat-latest"),
        help="OCI model name without the oci/ prefix.",
    )
    parser.add_argument(
        "--scenario",
        choices=("single", "multi", "both"),
        default="both",
        help="Which scenario to run.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of cProfile rows to print when profiling is enabled.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Optional path to dump cProfile stats.",
    )
    return parser.parse_args()


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _temperature_for_model(model: str) -> float | None:
    if model.startswith("openai.gpt-5"):
        return None
    return 0


def _build_llm(model: str) -> LLM:
    service_endpoint = os.getenv("OCI_TEST_SERVICE_ENDPOINT") or os.getenv(
        "OCI_SERVICE_ENDPOINT"
    )
    region = os.getenv("OCI_TEST_REGION") or os.getenv("OCI_REGION")
    if not service_endpoint and not region:
        raise SystemExit(
            "Set OCI_TEST_SERVICE_ENDPOINT/OCI_SERVICE_ENDPOINT or OCI_TEST_REGION/OCI_REGION."
        )

    kwargs: dict[str, Any] = {
        "model": f"oci/{model}",
        "compartment_id": _require_env("OCI_COMPARTMENT_ID"),
        "auth_type": os.getenv("OCI_AUTH_TYPE", "API_KEY"),
        "auth_profile": os.getenv("OCI_AUTH_PROFILE", "DEFAULT"),
        "auth_file_location": os.getenv("OCI_AUTH_FILE_LOCATION", "~/.oci/config"),
        "max_tokens": 1536,
        "temperature": _temperature_for_model(model),
    }
    if service_endpoint:
        kwargs["service_endpoint"] = service_endpoint
    return LLM(**kwargs)


def _build_agent(model: str) -> Agent:
    return Agent(
        role="Calculator",
        goal="Use tools to solve arithmetic problems accurately and consistently.",
        backstory="You are a precise calculator that must use the available tool.",
        llm=_build_llm(model),
        tools=[add_numbers],
        verbose=False,
    )


def _run_single(agent: Agent) -> ScenarioResult:
    started = time.perf_counter()
    result = agent.kickoff("Use add_numbers to calculate 20 + 22. Return only the final result.")
    elapsed_seconds = time.perf_counter() - started
    return ScenarioResult("single", elapsed_seconds, [result.raw])


def _run_multi(agent: Agent) -> ScenarioResult:
    prompts = [
        "Use add_numbers to calculate 2 + 5. Return only the final result.",
        "Use add_numbers to calculate 10 + 11. Return only the final result.",
        "Use add_numbers to calculate 20 + 22. Return only the final result.",
    ]
    started = time.perf_counter()
    responses = [agent.kickoff(prompt).raw for prompt in prompts]
    elapsed_seconds = time.perf_counter() - started
    return ScenarioResult("multi", elapsed_seconds, responses)


def _run_scenarios(model: str, scenario: str) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []
    if scenario in {"single", "both"}:
        results.append(_run_single(_build_agent(model)))
    if scenario in {"multi", "both"}:
        results.append(_run_multi(_build_agent(model)))
    return results


def _print_results(results: list[ScenarioResult]) -> None:
    for result in results:
        print(f"[{result.name}] elapsed={result.elapsed_seconds:.3f}s")
        for index, response in enumerate(result.responses, start=1):
            print(f"  response_{index}: {response}")


def main() -> None:
    args = _parse_args()
    profiler = cProfile.Profile()
    profiler.enable()
    results = _run_scenarios(args.model, args.scenario)
    profiler.disable()

    _print_results(results)

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(args.top)
    if args.profile_output:
        profiler.dump_stats(str(args.profile_output))
        print(f"cProfile stats written to {args.profile_output}")


if __name__ == "__main__":
    main()
