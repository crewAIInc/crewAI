from __future__ import annotations

from collections.abc import Iterable
import os
import re
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

from mem0 import Memory, MemoryClient  # type: ignore[import-untyped]

from crewai.memory.storage.interface import Storage
from crewai.rag.chromadb.utils import _sanitize_collection_name


if TYPE_CHECKING:
    from crewai.crew import Crew
    from crewai.utilities.types import LLMMessage, MessageRole


MAX_AGENT_ID_LENGTH_MEM0: Final[int] = 255
_ASSISTANT_MESSAGE_MARKER: Final[str] = "Final Answer:"
_USER_MESSAGE_PATTERN: Final[re.Pattern[str]] = re.compile(r"User message:\s*(.*)")


class BaseMetadata(TypedDict):
    short_term: Literal["short_term"]
    long_term: Literal["long_term"]
    entities: Literal["entity"]
    external: Literal["external"]


BASE_METADATA: Final[BaseMetadata] = {
    "short_term": "short_term",
    "long_term": "long_term",
    "entities": "entity",
    "external": "external",
}

MEMORY_TYPE_MAP: Final[dict[str, dict[str, str]]] = {
    "short_term": {"type": "short_term"},
    "long_term": {"type": "long_term"},
    "entities": {"type": "entity"},
    "external": {"type": "external"},
}


class BaseParams(TypedDict, total=False):
    """Parameters for Mem0 memory operations."""

    metadata: dict[str, Any]
    infer: bool
    includes: Any
    excludes: Any
    output_format: str
    version: str
    run_id: str
    user_id: str
    agent_id: str


class Mem0Config(TypedDict, total=False):
    """Configuration for Mem0Storage."""

    run_id: str
    includes: Any
    excludes: Any
    custom_categories: Any
    infer: bool
    api_key: str
    org_id: str
    project_id: str
    local_mem0_config: Any
    user_id: str
    agent_id: str


class Mem0Filter(TypedDict, total=False):
    """Filter dictionary for Mem0 search operations."""

    AND: list[dict[str, Any]]
    OR: list[dict[str, Any]]


class Mem0Storage(Storage):
    """
    Extends Storage to handle embedding and searching across entities using Mem0.
    """

    def __init__(
        self,
        type: Literal["short_term", "long_term", "entities", "external"],
        crew: Crew | None = None,
        config: Mem0Config | None = None,
    ) -> None:
        self.memory_type = type
        self.crew = crew
        if config is None:
            config = {}
        self.config: Mem0Config = config
        self.mem0_run_id = config.get("run_id")
        self.includes = config.get("includes")
        self.excludes = config.get("excludes")
        self.custom_categories = config.get("custom_categories")
        self.infer = config.get("infer", True)
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

    def _create_filter_for_search(self) -> Mem0Filter:
        """Create filter dictionary for search operations.

        Returns:
            Filter dictionary containing AND/OR conditions for querying data.
        """
        if self.memory_type == "short_term" and self.mem0_run_id:
            return {"AND": [{"run_id": self.mem0_run_id}]}

        user_id = self.config.get("user_id")
        agent_id = self.config.get("agent_id")
        if user_id and agent_id:
            return {"OR": [{"user_id": user_id}, {"agent_id": agent_id}]}
        if user_id:
            return {"AND": [{"user_id": user_id}]}
        if agent_id:
            return {"AND": [{"agent_id": agent_id}]}
        return {}

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        def _last_content(messages_: Iterable[LLMMessage], role: MessageRole) -> str:
            content = next(
                (
                    m.get("content", "")
                    for m in reversed(list(messages_))
                    if m.get("role") == role
                ),
                "",
            )
            return str(content) if content else ""

        conversations = []
        messages: Iterable[LLMMessage] = metadata.pop("messages", [])
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

        params: BaseParams = {
            "metadata": {"type": BASE_METADATA[self.memory_type], **metadata},
            "infer": self.infer,
        }

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
        params: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "version": "v2",
            "output_format": "v1.1",
        }

        if user_id := self.config.get("user_id", ""):
            params["user_id"] = user_id

        if self.memory_type in MEMORY_TYPE_MAP:
            params["metadata"] = MEMORY_TYPE_MAP[self.memory_type]
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

    def reset(self) -> None:
        if self.memory:
            self.memory.reset()

    @staticmethod
    def _sanitize_role(role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def _get_agent_name(self) -> str:
        if not self.crew:
            return ""

        agents = self.crew.agents
        agents_roles = "".join([self._sanitize_role(agent.role) for agent in agents])
        return _sanitize_collection_name(
            name=agents_roles, max_collection_length=MAX_AGENT_ID_LENGTH_MEM0
        )

    @staticmethod
    def _get_assistant_message(text: str) -> str:
        if _ASSISTANT_MESSAGE_MARKER in text:
            return text.split(_ASSISTANT_MESSAGE_MARKER, 1)[1].strip()
        return text

    @staticmethod
    def _get_user_message(text: str) -> str:
        match = _USER_MESSAGE_PATTERN.search(text)
        if match:
            return match.group(1).strip()
        return text
