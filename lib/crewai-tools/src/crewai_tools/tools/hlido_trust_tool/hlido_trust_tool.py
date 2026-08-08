"""Hlido trust tools for CrewAI.

Give a CrewAI agent independent, evidence-backed trust checks on other AI agents
before delegating work to them, using Hlido's public reviews API (https://hlido.eu).

* ``Hlido Trust Check`` — verdict (score, tier, PASS/FAIL gate) for a known agent slug.
* ``Hlido Recommend``   — find a Hlido-reviewed agent for a free-text need.

No API key required — both tools call the public Hlido API and add no new
dependencies beyond ``requests`` (already a CrewAI dependency).

Usage::

    from crewai_tools import HlidoTrustCheckTool, HlidoRecommendTool

    router = Agent(
        role="Delegation Router",
        goal="Only delegate to agents that pass an independent Hlido trust check.",
        tools=[HlidoTrustCheckTool(), HlidoRecommendTool()],
    )
"""

from __future__ import annotations

import json
from typing import Any

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

HLIDO_BASE_URL = "https://hlido.eu"

# Hlido tier bands (public): VITAL >= 90, STEADY >= 70, FADING >= 40, else FLATLINE.
_TIER_BANDS = [(90, "VITAL"), (70, "STEADY"), (40, "FADING")]


def _tier_for_score(score: float) -> str:
    for threshold, tier in _TIER_BANDS:
        if score >= threshold:
            return tier
    return "FLATLINE"


class HlidoTrustCheckInput(BaseModel):
    """Input for HlidoTrustCheckTool."""

    slug: str = Field(
        ...,
        description=(
            "Hlido agent slug to check, e.g. 'aider', 'crewai', 'langchain'. "
            "Lowercase and hyphenated — the identifier in a Hlido review URL "
            "(hlido.eu/reviews/<slug>/)."
        ),
    )
    min_score: float = Field(
        70.0,
        ge=0,
        le=100,
        description="Minimum acceptable Hlido score (0-100). 70 = STEADY tier or above.",
    )


class HlidoTrustCheckTool(BaseTool):
    """Independent trust verdict for a known AI agent, from Hlido's public reviews."""

    name: str = "Hlido Trust Check"
    description: str = (
        "Check an AI agent's independent Hlido trust verdict before relying on it. "
        "Given an agent slug (e.g. 'aider'), returns its Hlido score (0-100), tier "
        "(VITAL/STEADY/FADING/FLATLINE), a PASS/FAIL gate against min_score, and the "
        "reviewed strengths and red flags. Uses the public, evidence-backed Hlido API; "
        "no API key required."
    )
    args_schema: type[BaseModel] = HlidoTrustCheckInput
    base_url: str = HLIDO_BASE_URL
    timeout: int = 15

    def _run(self, slug: str, min_score: float = 70.0, **_: Any) -> str:
        slug = (slug or "").strip().lower()
        if not slug:
            return "Error: a Hlido agent slug is required (e.g. 'aider')."

        url = f"{self.base_url}/data/scorecards/{slug}.json"
        try:
            resp = requests.get(
                url, timeout=self.timeout, headers={"Accept": "application/json"}
            )
            if resp.status_code == 404:
                return (
                    f"No Hlido review found for '{slug}'. Hlido has not reviewed this "
                    "agent (or the slug differs). Use Hlido Recommend to find a "
                    "reviewed alternative."
                )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            return f"Error querying Hlido for '{slug}': {e}"
        except ValueError as e:
            return f"Error parsing Hlido response for '{slug}': {e}"

        score = data.get("score")
        if score is None:
            return f"Hlido returned no score for '{slug}'."

        resolved_slug = data.get("slug", slug)
        tier = data.get("tier") or _tier_for_score(score)
        passed = score >= min_score
        result = {
            "slug": resolved_slug,
            "name": data.get("name", slug),
            "score": score,
            "tier": tier,
            "min_score": min_score,
            "gate": "PASS" if passed else "FAIL",
            "verdict": "APPROVE" if passed else "REJECT",
            "strengths": data.get("what_it_does_well") or [],
            "red_flags": data.get("red_flags") or [],
            "review_url": f"{self.base_url}/reviews/{resolved_slug}/",
        }
        return json.dumps(result, ensure_ascii=False, indent=2)


class HlidoRecommendInput(BaseModel):
    """Input for HlidoRecommendTool."""

    need: str = Field(
        ...,
        description=(
            "Free-text description of what you need an agent for, "
            "e.g. 'AI coding assistant' or 'web scraping agent'."
        ),
    )
    min_score: float = Field(
        70.0,
        ge=0,
        le=100,
        description="Only recommend agents scoring at or above this (0-100).",
    )
    limit: int = Field(
        5, ge=1, le=25, description="Maximum number of agents to return."
    )


class HlidoRecommendTool(BaseTool):
    """Find Hlido-reviewed AI agents matching a free-text need, ranked by trust score."""

    name: str = "Hlido Recommend"
    description: str = (
        "Find independently-reviewed AI agents for a described need, ranked by Hlido "
        "trust score. Given a free-text need (e.g. 'AI coding agent'), returns matching "
        "agents from Hlido's public review registry with their score, tier and review "
        "URL. Uses the public Hlido API; no API key required."
    )
    args_schema: type[BaseModel] = HlidoRecommendInput
    base_url: str = HLIDO_BASE_URL
    timeout: int = 15

    def _run(self, need: str, min_score: float = 70.0, limit: int = 5, **_: Any) -> str:
        need = (need or "").strip().lower()
        if not need:
            return "Error: describe what you need an agent for."

        url = f"{self.base_url}/data/review-registry.json"
        try:
            resp = requests.get(
                url, timeout=self.timeout, headers={"Accept": "application/json"}
            )
            resp.raise_for_status()
            registry = resp.json()
        except requests.RequestException as e:
            return f"Error querying Hlido registry: {e}"
        except ValueError as e:
            return f"Error parsing Hlido registry: {e}"

        items = registry.get("items", []) if isinstance(registry, dict) else registry
        terms = [t for t in need.replace("/", " ").split() if len(t) > 2]

        matches: list[tuple] = []
        for item in items:
            if item.get("lane") != "reviewed":
                continue
            score = item.get("score")
            if score is None or score < min_score:
                continue
            haystack = " ".join(
                str(item.get(k, "")) for k in ("name", "slug", "category", "summary")
            ).lower()
            hits = sum(1 for t in terms if t in haystack)
            if hits:
                matches.append((hits, score, item))

        if not matches:
            return (
                f"No Hlido-reviewed agents scoring >= {min_score} matched '{need}'. "
                "Try a broader need or a lower min_score."
            )

        matches.sort(key=lambda m: (m[0], m[1]), reverse=True)
        out = []
        for _hits, score, item in matches[:limit]:
            slug = item.get("slug")
            out.append(
                {
                    "slug": slug,
                    "name": item.get("name", slug),
                    "score": score,
                    "tier": item.get("tier") or _tier_for_score(score),
                    "category": item.get("category"),
                    "review_url": f"{self.base_url}/reviews/{slug}/",
                }
            )
        return json.dumps(out, ensure_ascii=False, indent=2)
