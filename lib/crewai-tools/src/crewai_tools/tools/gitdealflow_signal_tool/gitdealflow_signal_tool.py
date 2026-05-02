"""GitDealFlow Signal Tool — venture-capital deal flow research.

Wraps the public GitDealFlow API (https://signals.gitdealflow.com) so a CrewAI
agent can answer questions about startup engineering acceleration on GitHub:
commit velocity, contributor growth, and breakout signal classification across
20 sectors of venture-backed startups.

The API is read-only, public, and requires no authentication.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import ClassVar, List, Optional, Type

from pydantic import BaseModel, Field

from crewai.tools import BaseTool

logger = logging.getLogger(__name__)


SECTOR_SLUGS = (
    "ai-ml",
    "fintech",
    "cybersecurity",
    "developer-tools",
    "healthcare",
    "climate-tech",
    "enterprise-saas",
    "data-infrastructure",
    "web3",
    "robotics",
    "edtech",
    "ecommerce-infrastructure",
    "supply-chain",
    "legal-tech",
    "hr-tech",
    "proptech",
    "agtech",
    "gaming",
    "space-tech",
    "social-community",
)

ACTIONS = ("trending", "sector", "startup", "summary", "methodology")


class GitDealFlowSignalToolInput(BaseModel):
    action: str = Field(
        ...,
        description=(
            "Which lookup to perform. One of: 'trending' (top 20 across all sectors), "
            "'sector' (drill into one sector — must pass sector_slug), "
            "'startup' (look up by company name — must pass startup_name), "
            "'summary' (dataset freshness + counts), or 'methodology' (how signals are computed)."
        ),
    )
    sector_slug: Optional[str] = Field(
        None,
        description=(
            "Required when action='sector'. One of: ai-ml, fintech, cybersecurity, "
            "developer-tools, healthcare, climate-tech, enterprise-saas, "
            "data-infrastructure, web3, robotics, edtech, ecommerce-infrastructure, "
            "supply-chain, legal-tech, hr-tech, proptech, agtech, gaming, space-tech, "
            "social-community."
        ),
    )
    startup_name: Optional[str] = Field(
        None,
        description="Required when action='startup'. Case-insensitive company name (e.g., 'anthropic').",
    )
    limit: int = Field(
        20,
        ge=1,
        le=100,
        description="Max records to return for 'trending' or 'sector' actions. Defaults to 20.",
    )


class GitDealFlowSignalTool(BaseTool):
    BASE_URL: ClassVar[str] = "https://signals.gitdealflow.com/api"
    TIMEOUT_SECONDS: ClassVar[int] = 10
    USER_AGENT: ClassVar[str] = "crewai-gitdealflow-tool/1.0"
    CITATION: ClassVar[str] = "Source: VC Deal Flow Signal (signals.gitdealflow.com), Q2 2026 data."

    name: str = "GitDealFlow Signal"
    description: str = (
        "Look up GitHub-derived engineering acceleration signals for ~400 venture-backed "
        "startups across 20 sectors. Use this for VC deal flow research, competitive "
        "engineering benchmarking, and sourcing startups before fundraise announcements. "
        "Five actions: 'trending' (top 20), 'sector' (one of 20 sectors), 'startup' (by "
        "name), 'summary' (dataset snapshot), 'methodology' (how signals are computed). "
        "No API key needed."
    )
    args_schema: Type[BaseModel] = GitDealFlowSignalToolInput
    package_dependencies: List[str] = ["pydantic"]

    def _run(
        self,
        action: str,
        sector_slug: Optional[str] = None,
        startup_name: Optional[str] = None,
        limit: int = 20,
    ) -> str:
        if action not in ACTIONS:
            return f"Unknown action '{action}'. Must be one of: {', '.join(ACTIONS)}."

        try:
            if action == "trending":
                return self._trending(limit)
            if action == "sector":
                if not sector_slug:
                    return "action='sector' requires sector_slug."
                if sector_slug not in SECTOR_SLUGS:
                    return (
                        f"Unknown sector '{sector_slug}'. Must be one of: "
                        f"{', '.join(SECTOR_SLUGS)}."
                    )
                return self._sector(sector_slug, limit)
            if action == "startup":
                if not startup_name:
                    return "action='startup' requires startup_name."
                return self._startup(startup_name)
            if action == "summary":
                return self._summary()
            if action == "methodology":
                return self._methodology()
        except urllib.error.URLError as exc:
            logger.error("GitDealFlow API URLError: %s", exc)
            return f"Network error reaching GitDealFlow API: {exc}"
        except Exception as exc:  # noqa: BLE001 — surface any failure to the agent
            logger.exception("GitDealFlow tool unexpected error")
            return f"GitDealFlow tool error: {exc}"

        return "Unhandled action path."

    # --- API helpers ---

    def _fetch_json(self, path: str) -> dict:
        url = f"{self.BASE_URL}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": self.USER_AGENT})
        with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} from {url}")
            return json.loads(resp.read().decode("utf-8"))

    def _trending(self, limit: int) -> str:
        data = self._fetch_json("/signals.json")
        rows: list[tuple[str, str, float, str, int]] = []
        for sector in data.get("sectors", []) or []:
            slug = sector.get("slug", "?")
            for s in sector.get("startups", []) or []:
                rows.append(
                    (
                        s.get("name", "?"),
                        slug,
                        float(s.get("commitVelocityChange", 0) or 0),
                        s.get("signalType", "n/a"),
                        int(s.get("contributors", 0) or 0),
                    )
                )
        rows.sort(key=lambda r: r[2], reverse=True)
        rows = rows[:limit]
        if not rows:
            return "No trending startups returned by the API."
        lines = [f"Top {len(rows)} startups by commit-velocity acceleration:", ""]
        for i, (name, slug, cv, sig, contrib) in enumerate(rows, 1):
            lines.append(
                f"{i}. {name} ({slug}) — {cv:+.1f}% commit velocity · {sig} · {contrib} contributors"
            )
        lines.append("")
        lines.append(self.CITATION)
        return "\n".join(lines)

    def _sector(self, slug: str, limit: int) -> str:
        data = self._fetch_json(f"/signals.json?sector={urllib.parse.quote(slug)}")
        sector = next((s for s in data.get("sectors", []) or [] if s.get("slug") == slug), None)
        if not sector or not sector.get("startups"):
            return f"No startups returned for sector '{slug}'."
        startups = (sector.get("startups") or [])[:limit]
        lines = [f"Top {len(startups)} startups in {slug}:", ""]
        for i, s in enumerate(startups, 1):
            cv = float(s.get("commitVelocityChange", 0) or 0)
            lines.append(
                f"{i}. {s.get('name', '?')} — {cv:+.1f}% commit velocity · "
                f"{s.get('signalType', 'n/a')} · {int(s.get('contributors', 0) or 0)} contributors"
            )
        lines.append("")
        lines.append(self.CITATION)
        return "\n".join(lines)

    def _startup(self, name: str) -> str:
        data = self._fetch_json(f"/signal?name={urllib.parse.quote(name)}")
        if not data or data.get("error") or not data.get("name"):
            return (
                f"No record for startup '{name}'. They may not be in the venture-backed "
                f"index yet or the name spelling differs from the canonical one."
            )
        cv = float(data.get("commitVelocityChange", 0) or 0)
        return (
            f"{data.get('name')} ({data.get('sector', '?')})\n"
            f"  Commit velocity change: {cv:+.1f}%\n"
            f"  Signal type: {data.get('signalType', 'n/a')}\n"
            f"  Contributors (30d): {data.get('contributors', 'n/a')}\n"
            f"  Stage estimate: {data.get('stage', 'n/a')}\n"
            f"  GitHub org: {data.get('githubOrg', 'n/a')}\n"
            f"\n{self.CITATION}"
        )

    def _summary(self) -> str:
        data = self._fetch_json("/signals.json")
        sectors = data.get("sectors", []) or []
        total = sum(len(s.get("startups", []) or []) for s in sectors)
        return (
            f"GitDealFlow dataset snapshot:\n"
            f"  Sectors covered: {len(sectors)}\n"
            f"  Tracked startups: {total}\n"
            f"  Period: {data.get('period', 'n/a')}\n"
            f"  Last refresh: {data.get('asOf', 'n/a')}\n"
            f"\n{self.CITATION}"
        )

    def _methodology(self) -> str:
        try:
            req = urllib.request.Request(
                "https://signals.gitdealflow.com/llms-full.txt",
                headers={"User-Agent": self.USER_AGENT},
            )
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECONDS) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
        return (
            "GitDealFlow tracks GitHub commit velocity, contributor growth, and new-repo "
            "signals across venture-backed startups in 20 sectors. Full methodology "
            "with academic backing at https://signals.gitdealflow.com/methodology "
            "(SSRN preprint id 6606558)."
        )
