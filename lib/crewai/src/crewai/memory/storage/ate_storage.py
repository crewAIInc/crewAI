"""
CrewAI Memory Storage integration for FoodforThought's .mv2 format.

Uses the ``ate`` CLI (https://github.com/kindlyrobotics/monorepo) to store
and retrieve memories in portable .mv2 files — encrypted, offline-first,
and compatible with any embedding provider.

Install the ate CLI::

    pip install ate-robotics

Usage::

    from crewai_ate_storage import AteStorage

    crew = Crew(
        agents=[...],
        tasks=[...],
        memory=True,
        long_term_memory=AteStorage(type="long_term"),
        short_term_memory=AteStorage(type="short_term"),
        entity_memory=AteStorage(type="entities"),
    )
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from crewai.memory.storage.interface import Storage

logger = logging.getLogger(__name__)


class AteStorage(Storage):
    """CrewAI-compatible memory storage backed by FoodforThought .mv2 files.

    The ``ate`` CLI must be installed and available on ``$PATH``
    (``pip install ate-robotics``).

    Parameters
    ----------
    type:
        Memory type — one of ``"short_term"``, ``"long_term"``,
        ``"entities"``, or ``"external"``.
    crew:
        Optional crew instance (used to derive a default storage path).
    config:
        Optional configuration dict. Recognised keys:

        * ``memory_path`` – explicit path to the ``.mv2`` file.
        * ``embedding_provider`` – reserved for future use.
    """

    VALID_TYPES: set[str] = {"short_term", "long_term", "entities", "external"}

    def __init__(
        self,
        type: str,
        crew: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if type not in self.VALID_TYPES:
            raise ValueError(
                f"Invalid memory type {type!r}. "
                f"Must be one of {sorted(self.VALID_TYPES)}."
            )

        self.type: str = type
        self.crew = crew
        self.config: dict[str, Any] = config or {}

        self._memory_path: str = self._resolve_memory_path()
        self._initialized: bool = False

        self._check_ate_cli()

    # ------------------------------------------------------------------
    # Public API (Storage interface)
    # ------------------------------------------------------------------

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """Persist a memory entry to the .mv2 file."""
        self._ensure_initialized()

        title: str = str(metadata.get("agent", "unknown"))
        tags: list[str] = [self.type]
        if "tags" in metadata:
            extra = metadata["tags"]
            if isinstance(extra, str):
                tags.append(extra)
            elif isinstance(extra, (list, tuple)):
                tags.extend(str(t) for t in extra)

        cmd: list[str] = [
            "ate", "memory", "add", self._memory_path,
            "--text", str(value),
            "--tags", ",".join(tags),
            "--title", title,
            "--format", "json",
        ]

        result = self._run(cmd)
        if result.returncode != 0:
            logger.error(
                "ate memory add failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search the .mv2 file for memories matching *query*.

        Returns a list of dicts with ``content``, ``metadata``, and
        ``score`` keys — matching CrewAI's expected format.
        """
        self._ensure_initialized()

        cmd: list[str] = [
            "ate", "memory", "search", self._memory_path,
            "--query", query,
            "--top-k", str(limit),
            "--format", "json",
        ]

        result = self._run(cmd)
        if result.returncode != 0:
            logger.error(
                "ate memory search failed (rc=%d): %s",
                result.returncode,
                result.stderr.strip(),
            )
            return []

        return self._parse_search_results(result.stdout, score_threshold)

    def reset(self) -> None:
        """Re-initialise the .mv2 file, discarding all stored memories."""
        path = Path(self._memory_path)
        if path.exists():
            path.unlink()
        self._initialized = False
        self._init_memory_file()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_memory_path(self) -> str:
        explicit: str | None = self.config.get("memory_path")
        if explicit:
            return str(Path(explicit).expanduser())

        crew_name: str = "default"
        if self.crew is not None:
            crew_name = getattr(self.crew, "name", None) or "default"
            crew_name = crew_name.replace(" ", "_").lower()

        return str(
            Path.home() / ".ate" / "memories" / crew_name / f"{self.type}.mv2"
        )

    def _check_ate_cli(self) -> None:
        if shutil.which("ate") is None:
            raise FileNotFoundError(
                "The 'ate' CLI was not found on $PATH. "
                "Install it with: pip install ate-robotics"
            )

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        path = Path(self._memory_path)
        if not path.exists():
            self._init_memory_file()
        self._initialized = True

    def _init_memory_file(self) -> None:
        path = Path(self._memory_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            "ate", "memory", "init", self._memory_path,
            "--format", "json",
        ]

        result = self._run(cmd)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to initialise .mv2 file at {self._memory_path}: "
                f"{result.stderr.strip()}"
            )
        self._initialized = True
        logger.info("Initialised .mv2 memory file at %s", self._memory_path)

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(cmd, capture_output=True, text=True)

    @staticmethod
    def _parse_search_results(
        raw_output: str,
        score_threshold: float,
    ) -> list[dict[str, Any]]:
        if not raw_output.strip():
            return []
        try:
            data = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.error("Failed to parse ate search output as JSON.")
            return []

        raw_results: list[dict[str, Any]] = (
            data if isinstance(data, list) else data.get("results", [])
        )

        results: list[dict[str, Any]] = []
        for entry in raw_results:
            score: float = float(entry.get("score", 0.0))
            if score < score_threshold:
                continue
            results.append({
                "content": entry.get("text", ""),
                "metadata": {
                    k: v for k, v in entry.items()
                    if k not in ("text", "score")
                },
                "score": score,
            })
        return results
