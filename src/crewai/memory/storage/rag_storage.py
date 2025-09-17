import logging
import warnings
from typing import Any

from crewai.rag.chromadb.config import ChromaDBConfig
from crewai.rag.config.utils import get_rag_client
from crewai.rag.core.base_client import BaseClient
from crewai.rag.embeddings.factory import get_embedding_function
from crewai.rag.factory import create_client
from crewai.rag.storage.base_rag_storage import BaseRAGStorage
from crewai.rag.types import BaseRecord
from crewai.utilities.constants import MAX_FILE_NAME_LENGTH
from crewai.utilities.paths import db_storage_path


class RAGStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    def __init__(
        self, type, allow_reset=True, embedder_config=None, crew=None, path=None
    ):
        super().__init__(type, allow_reset, embedder_config, crew)
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents
        self.storage_file_name = self._build_storage_file_name(type, agents)

        self.type = type
        self._client: BaseClient | None = None

        self.allow_reset = allow_reset
        self.path = path

        warnings.filterwarnings(
            "ignore",
            message=r".*'model_fields'.*is deprecated.*",
            module=r"^chromadb(\.|$)",
        )

        if self.embedder_config:
            embedding_function = get_embedding_function(self.embedder_config)
            config = ChromaDBConfig(embedding_function=embedding_function)
            self._client = create_client(config)

    def _get_client(self) -> BaseClient:
        """Get the appropriate client - instance-specific or global."""
        return self._client if self._client else get_rag_client()

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def _build_storage_file_name(self, type: str, file_name: str) -> str:
        """
        Ensures file name does not exceed max allowed by OS
        """
        base_path = f"{db_storage_path()}/{type}"

        if len(file_name) > MAX_FILE_NAME_LENGTH:
            logging.warning(
                f"Trimming file name from {len(file_name)} to {MAX_FILE_NAME_LENGTH} characters."
            )
            file_name = file_name[:MAX_FILE_NAME_LENGTH]

        return f"{base_path}/{file_name}"

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        try:
            client = self._get_client()
            collection_name = (
                f"memory_{self.type}_{self.agents}"
                if self.agents
                else f"memory_{self.type}"
            )
            client.get_or_create_collection(collection_name=collection_name)

            document: BaseRecord = {"content": value}
            if metadata:
                document["metadata"] = metadata

            client.add_documents(collection_name=collection_name, documents=[document])
        except Exception as e:
            logging.error(f"Error during {self.type} save: {e!s}")

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: dict[str, Any] | None = None,
        score_threshold: float = 0.35,
    ) -> list[Any]:
        try:
            client = self._get_client()
            collection_name = (
                f"memory_{self.type}_{self.agents}"
                if self.agents
                else f"memory_{self.type}"
            )
            return client.search(
                collection_name=collection_name,
                query=query,
                limit=limit,
                metadata_filter=filter,
                score_threshold=score_threshold,
            )
        except Exception as e:
            logging.error(f"Error during {self.type} search: {e!s}")
            return []

    def reset(self) -> None:
        try:
            client = self._get_client()
            collection_name = (
                f"memory_{self.type}_{self.agents}"
                if self.agents
                else f"memory_{self.type}"
            )
            client.delete_collection(collection_name=collection_name)
        except Exception as e:
            if "attempt to write a readonly database" in str(
                e
            ) or "does not exist" in str(e):
                # Ignore readonly database and collection not found errors (already reset)
                pass
            else:
                raise Exception(
                    f"An error occurred while resetting the {self.type} memory: {e}"
                ) from e
