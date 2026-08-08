import os
from typing import List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field


class MengramSearchToolSchema(BaseModel):
    """Input for MengramSearchTool: a natural-language memory query."""

    query: str = Field(
        description="What to look for, e.g. 'database preferences' or 'last deploy incident'"
    )


class MengramSaveToolSchema(BaseModel):
    """Input for MengramSaveTool: the fact/decision to persist."""

    content: str = Field(
        description="The durable fact, preference, event, or decision to remember"
    )


class MengramProceduresToolSchema(BaseModel):
    """Input for MengramProceduresTool: the task to fetch learned workflows for."""

    task: str = Field(description="The task at hand, e.g. 'deploy to production'")


class _MengramBaseTool(BaseTool):
    """Shared config + client for the Mengram memory tools (https://mengram.io).

    api_key defaults to the MENGRAM_API_KEY env var, resolved at call time (not
    import time) so it picks up the environment as configured at run.
    """

    api_key: Optional[str] = None
    user_id: str = "default"
    base_url: Optional[str] = None
    limit: int = 5
    package_dependencies: List[str] = ["mengram-ai"]
    env_vars: List[EnvVar] = [
        EnvVar(
            name="MENGRAM_API_KEY",
            description="API key for Mengram memory (free tier at https://mengram.io)",
            required=True,
        ),
    ]

    def _resolve_key(self) -> Optional[str]:
        # Explicit key wins; otherwise read the env var at call time.
        return self.api_key or os.getenv("MENGRAM_API_KEY")

    def _client(self):
        try:
            from mengram import Mengram  # type: ignore
        except ImportError:
            raise ImportError(
                "`mengram-ai` package not found, please run `uv add mengram-ai`"
            )
        key = self._resolve_key()
        if not key:
            raise EnvironmentError(
                "MENGRAM_API_KEY environment variable or api_key= is required"
            )
        self.api_key = key
        if self.base_url:
            return Mengram(api_key=self.api_key, base_url=self.base_url)
        return Mengram(api_key=self.api_key)


class MengramSearchTool(_MengramBaseTool):
    """Semantic + episodic + procedural memory search over the user's Mengram vault."""

    name: str = "Mengram memory search"
    description: str = (
        "Search the user's long-term memory for facts, preferences, past events, "
        "and learned workflows. Use before answering anything that may depend on "
        "who the user is or what happened before."
    )
    args_schema: Type[BaseModel] = MengramSearchToolSchema

    def _run(self, query: str) -> str:
        # Unified search across all three memory types, not facts-only.
        res = self._client().search_all(query, user_id=self.user_id, limit=self.limit)
        lines = []
        for r in res.get("semantic", []):
            facts = "; ".join(r.get("facts", [])[:5])
            lines.append(f"fact — {r.get('entity', '?')} ({r.get('type', '?')}): {facts}")
        for e in res.get("episodic", []):
            lines.append(f"event — {e.get('summary', '')}")
        for p in res.get("procedural", []):
            lines.append(f"procedure — {p.get('name', '?')} (v{p.get('version', 1)})")
        return "\n".join(lines) if lines else "No memories found for this query."


class MengramSaveTool(_MengramBaseTool):
    """Persist a durable fact/decision to the user's long-term Mengram memory."""

    name: str = "Mengram memory save"
    description: str = (
        "Save an important new fact, preference, event, or decision to the user's "
        "long-term memory so future sessions remember it. Use sparingly for "
        "durable information, not small talk."
    )
    args_schema: Type[BaseModel] = MengramSaveToolSchema

    def _run(self, content: str) -> str:
        res = self._client().add_text(content, user_id=self.user_id, source="crewai")
        return f"Saved (status: {res.get('status', 'ok')})."


class MengramProceduresTool(_MengramBaseTool):
    """Retrieve learned workflows (with track record + preconditions) for a task."""

    name: str = "Mengram learned procedures"
    description: str = (
        "Retrieve the user's learned workflows (procedures) relevant to a task — "
        "step-by-step playbooks that evolved from past successes and failures, "
        "with preconditions to verify. Use before performing a multi-step task "
        "the user has likely done before (deploys, releases, setups)."
    )
    args_schema: Type[BaseModel] = MengramProceduresToolSchema

    def _run(self, task: str) -> str:
        procedures = self._client().procedures(
            query=task, limit=self.limit, user_id=self.user_id
        )
        if not procedures:
            return "No learned procedures for this task yet."
        lines = []
        for p in procedures:
            steps = "; ".join(
                f"{s.get('step', '?')}. {s.get('action', '')}"
                for s in p.get("steps", [])
            )
            preconditions = (
                (p.get("metadata") or {}).get("preconditions")
                or p.get("preconditions")
                or []
            )
            pre = f" Verify first: {preconditions}" if preconditions else ""
            lines.append(
                f"{p.get('name', '?')} (v{p.get('version', 1)}, "
                f"{p.get('success_count', 0)} successes / "
                f"{p.get('fail_count', 0)} failures): {steps}.{pre}"
            )
        return "\n".join(lines)
