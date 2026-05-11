"""Ejentum Reasoning Harness tool for CrewAI.

Exposes the four Ejentum cognitive harnesses (reasoning, code, anti-deception,
memory) as a single agent-callable tool. Each call retrieves a task-matched
cognitive scaffold from a library of 679 cognitive operations engineered in
natural language. The agent ingests the scaffold (failure pattern, executable
procedure, suppression vectors, falsification test) and writes from it.

Free tier: 100 calls, no card, at https://ejentum.com/pricing.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from crewai.tools import BaseTool, EnvVar
from pydantic import Field

from crewai_tools.tools.ejentum_reasoning_harness_tool.schemas import (
    EjentumHarnessParams,
)


DEFAULT_API_URL = "https://ejentum-main-ab125c3.zuplo.app/logicv1/"


class EjentumHarnessTool(BaseTool):
    """Call one of the four Ejentum cognitive harnesses to retrieve a task-matched scaffold.

    Use `harness_reasoning` (`mode='reasoning'`) before analytical, diagnostic,
    planning, or multi-step tasks. Use `mode='code'` before producing or
    reviewing code. Use `mode='anti-deception'` when a prompt pressures the
    agent to validate, certify, or soften an honest assessment. Use
    `mode='memory'` only when sharpening an observation already formed about
    cross-turn drift.

    Requires the `EJENTUM_API_KEY` environment variable. Free tier (100 calls,
    no card) at https://ejentum.com/pricing.
    """

    name: str = "Ejentum Reasoning Harness"
    description: str = (
        "Retrieve a task-matched cognitive scaffold from Ejentum's library of "
        "679 cognitive operations engineered in natural language. Pass the "
        "task you are about to work on and pick one of four modes "
        "('reasoning', 'code', 'anti-deception', 'memory'). Returns a "
        "structured scaffold (failure pattern, executable procedure, "
        "suppression vectors, falsification test) the agent ingests before "
        "responding."
    )
    args_schema: type = EjentumHarnessParams
    api_url: str = DEFAULT_API_URL
    timeout_seconds: float = 10.0
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="EJENTUM_API_KEY",
                description=(
                    "API key for the Ejentum Logic API. Free tier (100 calls, "
                    "no card) at https://ejentum.com/pricing."
                ),
                required=True,
            ),
        ]
    )

    def _run(self, **kwargs: Any) -> str:
        raw_query = kwargs.get("query")
        query = raw_query.strip() if isinstance(raw_query, str) else ""
        mode = kwargs.get("mode")

        if not query:
            return "Ejentum harness call failed: 'query' is required."
        if mode not in {"reasoning", "code", "anti-deception", "memory"}:
            return (
                f"Ejentum harness call failed: 'mode' must be one of "
                f"reasoning|code|anti-deception|memory, got '{mode}'."
            )

        api_key = os.environ.get("EJENTUM_API_KEY")
        if not api_key:
            return (
                "Ejentum harness call failed: EJENTUM_API_KEY environment "
                "variable is not set. Get a free key (100 calls, no card) at "
                "https://ejentum.com/pricing."
            )

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"query": query, "mode": mode},
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            return f"Ejentum harness call failed: network error: {exc}"

        if response.status_code == 401:
            return (
                "Ejentum harness call failed: unauthorized (401). Check the "
                "EJENTUM_API_KEY value. Get a key at https://ejentum.com/pricing."
            )
        if response.status_code != 200:
            return (
                f"Ejentum harness call failed: HTTP {response.status_code}. "
                f"Response: {response.text[:300]}"
            )

        try:
            data = response.json()
        except ValueError:
            return (
                f"Ejentum harness call failed: response is not valid JSON. "
                f"Body: {response.text[:300]}"
            )

        # The API returns: [{<mode>: <scaffold_string>}]
        if isinstance(data, list) and data and isinstance(data[0], dict):
            scaffold = data[0].get(mode)
            if scaffold:
                return scaffold

        return (
            f"Ejentum harness call returned an unexpected response shape: "
            f"{str(data)[:300]}"
        )
