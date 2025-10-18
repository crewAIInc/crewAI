import os
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from mem0 import Memory, MemoryClient  # type: ignore[import-untyped,import-not-found]

from crewai.memory.storage.interface import Storage
from crewai.rag.chromadb.utils import _sanitize_collection_name

MAX_AGENT_ID_LENGTH_MEM0 = 255


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    """

    def __init__(self, type, crew=None, config=None):
        super().__init__()

        self._validate_type(type)
        self.memory_type = type
        self.crew = crew
        self.config = config or {}

        self._extract_config_values()
        self._initialize_memory()

    def _validate_type(self, type):
        supported_types = {"short_term", "long_term", "entities", "external"}
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for Mem0Storage. "
                f"Must be one of: {', '.join(supported_types)}"
            )

    def _extract_config_values(self):
        self.mem0_run_id = self.config.get("run_id")
        self.includes = self.config.get("includes")
        self.excludes = self.config.get("excludes")
        self.custom_categories = self.config.get("custom_categories")
        self.infer = self.config.get("infer", True)

    def _initialize_memory(self):
        api_key = self.config.get("api_key") or os.getenv("MEM0_API_KEY")
        org_id = self.config.get("org_id")
        project_id = self.config.get("project_id")
        local_config = self.config.get("local_mem0_config")

        if api_key:
            self.memory = (
                MemoryClient(api_key=api_key, org_id=org_id, project_id=project_id)
                if org_id and project_id
                else MemoryClient(api_key=api_key)
            )
            if self.custom_categories:
                self.memory.update_project(custom_categories=self.custom_categories)
        else:
            self.memory = (
                Memory.from_config(local_config)
                if local_config and len(local_config)
                else Memory()
            )

    def _create_filter_for_search(self):
        """
        Returns:
            dict: A filter dictionary containing AND conditions for querying data.
                - Includes user_id and agent_id if both are present.
                - Includes user_id if only user_id is present.
                - Includes agent_id if only agent_id is present.
                - Includes run_id if memory_type is 'short_term' and
                  mem0_run_id is present.
        """
        filter = defaultdict(list)

        if self.memory_type == "short_term" and self.mem0_run_id:
            filter["AND"].append({"run_id": self.mem0_run_id})
        else:
            user_id = self.config.get("user_id", "")
            agent_id = self.config.get("agent_id", "")

            if user_id and agent_id:
                filter["OR"].append({"user_id": user_id})
                filter["OR"].append({"agent_id": agent_id})
            elif user_id:
                filter["AND"].append({"user_id": user_id})
            elif agent_id:
                filter["AND"].append({"agent_id": agent_id})

        return filter

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        def _last_content(messages: Iterable[dict[str, Any]], role: str) -> str:
            return next(
                (
                    m.get("content", "")
                    for m in reversed(list(messages))
                    if m.get("role") == role
                ),
                "",
            )

        conversations = []
        messages = metadata.pop("messages", None)
        if messages:
            last_user = _last_content(messages, "user")
            last_assistant = _last_content(messages, "assistant")

            if user_msg := self._get_user_message(last_user):
                conversations.append({"role": "user", "content": user_msg})

            if assistant_msg := self._get_assistant_message(last_assistant):
                conversations.append({"role": "assistant", "content": assistant_msg})
        else:
            conversations.append({"role": "assistant", "content": value})

        user_id = self.config.get("user_id", "")

        base_metadata = {
            "short_term": "short_term",
            "long_term": "long_term",
            "entities": "entity",
            "external": "external",
        }

        # Shared base params
        params: dict[str, Any] = {
            "metadata": {"type": base_metadata[self.memory_type], **metadata},
            "infer": self.infer,
        }

        # MemoryClient-specific overrides
        if isinstance(self.memory, MemoryClient):
            params["includes"] = self.includes
            params["excludes"] = self.excludes
            params["output_format"] = "v1.1"
            params["version"] = "v2"

        if self.memory_type == "short_term" and self.mem0_run_id:
            params["run_id"] = self.mem0_run_id

        if user_id:
            params["user_id"] = user_id

        if agent_id := self.config.get("agent_id", self._get_agent_name()):
            params["agent_id"] = agent_id

        self.memory.add(conversations, **params)

    def search(
        self, query: str, limit: int = 5, score_threshold: float = 0.6
    ) -> list[Any]:
        params = {
            "query": query,
            "limit": limit,
            "version": "v2",
            "output_format": "v1.1",
        }

        if user_id := self.config.get("user_id", ""):
            params["user_id"] = user_id

        memory_type_map = {
            "short_term": {"type": "short_term"},
            "long_term": {"type": "long_term"},
            "entities": {"type": "entity"},
            "external": {"type": "external"},
        }

        if self.memory_type in memory_type_map:
            params["metadata"] = memory_type_map[self.memory_type]
            if self.memory_type == "short_term":
                params["run_id"] = self.mem0_run_id

        # Discard the filters for now since we create the filters
        # automatically when the crew is created.

        params["filters"] = self._create_filter_for_search()
        params["threshold"] = score_threshold

        if isinstance(self.memory, Memory):
            del params["metadata"], params["version"], params["output_format"]
            if params.get("run_id"):
                del params["run_id"]

        results = self.memory.search(**params)

        # This makes it compatible for Contextual Memory to retrieve
        for result in results["results"]:
            result["content"] = result["memory"]

        return [r for r in results["results"]]

    def reset(self):
        if self.memory:
            self.memory.reset()

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def _get_agent_name(self) -> str:
        if not self.crew:
            return ""

        agents = self.crew.agents
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        return _sanitize_collection_name(
            name=agents, max_collection_length=MAX_AGENT_ID_LENGTH_MEM0
        )

    def _get_assistant_message(self, text: str) -> str:
        marker = "Final Answer:"
        if marker in text:
            return text.split(marker, 1)[1].strip()
        return text

    def _get_user_message(self, text: str) -> str:
        pattern = r"User message:\s*(.*)"
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        return text
