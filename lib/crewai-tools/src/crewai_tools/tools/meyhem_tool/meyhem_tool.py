from __future__ import annotations
from builtins import type as type_
import json
import httpx
from crewai.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field


class MeyhemSearchSchema(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Maximum number of results to return")
    include_content: bool = Field(False, description="Include page content in results")
    freshness: str | None = Field(
        None, description="Cache freshness: realtime, hour, day, week"
    )


class MeyhemDiscoverSchema(BaseModel):
    task: str = Field(..., description="What capability do you need?")
    max_results: int = Field(5, description="Maximum number of results")
    ecosystem: str | None = Field(
        None, description="Filter by ecosystem: mcp, openclaw, or None for all"
    )


class MeyhemSearchTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "MeyhemSearchTool"
    description: str = "AI-powered web search optimized for agents. Returns ranked results with optional page content. No API key required."
    args_schema: type_[BaseModel] = MeyhemSearchSchema
    base_url: str = "https://api.rhdxm.com"
    agent_id: str = "crewai-agent"
    package_dependencies: list[str] = Field(default_factory=lambda: ["httpx"])

    def _run(
        self,
        query: str,
        max_results: int = 10,
        include_content: bool = False,
        freshness: str | None = None,
    ) -> str:
        payload = dict(
            query=query,
            max_results=max_results,
            include_content=include_content,
            agent_id=self.agent_id,
        )
        if freshness:
            payload["freshness"] = freshness
        try:
            r = httpx.post(f"{self.base_url}/search", json=payload, timeout=30)
            r.raise_for_status()
            return json.dumps(r.json()["results"], indent=2)
        except Exception as e:
            return f"Search failed: {e}"


class MeyhemDiscoverTool(BaseTool):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = "MeyhemDiscoverTool"
    description: str = "Find the best MCP server or OpenClaw skill for a given task. Searches 22,000+ capabilities across ecosystems. No API key required."
    args_schema: type_[BaseModel] = MeyhemDiscoverSchema
    base_url: str = "https://api.rhdxm.com"
    package_dependencies: list[str] = Field(default_factory=lambda: ["httpx"])

    def _run(
        self, task: str, max_results: int = 5, ecosystem: str | None = None
    ) -> str:
        payload = dict(task=task, max_results=max_results)
        if ecosystem:
            payload["ecosystem"] = ecosystem
        try:
            r = httpx.post(f"{self.base_url}/find-capability", json=payload, timeout=30)
            r.raise_for_status()
            return json.dumps(r.json()["results"], indent=2)
        except Exception as e:
            return f"Discovery failed: {e}"
