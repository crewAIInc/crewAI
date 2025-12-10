from __future__ import annotations

import os
from typing import Any

from crewai.memory.storage.interface import Storage
from crewai.utilities.paths import get_crewai_storage_dir


class OpenMemoryStorage(Storage):
    """
    Extends Storage to handle embedding and searching using OpenMemory.

    OpenMemory is a local-first persistent memory engine for AI applications.
    It provides cognitive memory capabilities without requiring external vector databases.
    """

    def __init__(
        self,
        type: str,
        crew: Any | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self._validate_type(type)
        self.memory_type = type
        self.crew = crew
        self.config = config or {}

        self._extract_config_values()
        self._initialize_memory()

    def _validate_type(self, type: str) -> None:
        supported_types = {"short_term", "long_term", "entities", "external"}
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for OpenMemoryStorage. "
                f"Must be one of: {', '.join(supported_types)}"
            )

    def _extract_config_values(self) -> None:
        self.user_id = self.config.get("user_id")
        self.path = self.config.get("path")
        self.tier = self.config.get("tier", "fast")
        self.embeddings = self.config.get(
            "embeddings", {"provider": "synthetic"}
        )

        if not self.path:
            storage_dir = get_crewai_storage_dir()
            self.path = os.path.join(
                storage_dir, f"openmemory_{self.memory_type}.sqlite"
            )

    def _initialize_memory(self) -> None:
        try:
            from openmemory import (
                OpenMemory,  # type: ignore[import-untyped,import-not-found]
            )
        except ImportError as e:
            raise ImportError(
                "OpenMemory is not installed. "
                "Please install it with `pip install openmemory-py`."
            ) from e

        os.makedirs(os.path.dirname(self.path), exist_ok=True)

        self.memory = OpenMemory(
            mode="local",
            path=self.path,
            tier=self.tier,
            embeddings=self.embeddings,
        )

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """
        Save a memory item to OpenMemory.

        Args:
            value: The content to save.
            metadata: Additional metadata to associate with the memory.
        """
        params: dict[str, Any] = {}

        if self.user_id:
            params["userId"] = self.user_id

        tags = metadata.pop("tags", None)
        if tags:
            params["tags"] = tags

        params["metadata"] = {
            "type": self.memory_type,
            **metadata,
        }

        self.memory.add(str(value), **params)

    def search(
        self, query: str, limit: int = 5, score_threshold: float = 0.6
    ) -> list[Any]:
        """
        Search for relevant memories in OpenMemory.

        Args:
            query: The search query.
            limit: Maximum number of results to return.
            score_threshold: Minimum similarity score for results.

        Returns:
            List of matching memory entries with 'content' field.
        """
        params: dict[str, Any] = {
            "k": limit,
        }

        filters: dict[str, Any] = {}
        if self.user_id:
            filters["user_id"] = self.user_id

        if filters:
            params["filters"] = filters

        results = self.memory.query(query, **params)

        normalized_results = []
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    normalized = dict(result)
                    if "content" not in normalized and "memory" in normalized:
                        normalized["content"] = normalized["memory"]
                    elif "content" not in normalized:
                        normalized["content"] = str(result)

                    score = normalized.get("score", 1.0)
                    if score >= score_threshold:
                        normalized_results.append(normalized)
                else:
                    normalized_results.append({"content": str(result)})

        return normalized_results

    def reset(self) -> None:
        """
        Reset the OpenMemory storage by closing and deleting the database file.
        """
        if hasattr(self, "memory") and self.memory:
            try:
                self.memory.close()
            except Exception:
                self.memory = None

        if self.path and os.path.exists(self.path):
            os.remove(self.path)

        self._initialize_memory()
