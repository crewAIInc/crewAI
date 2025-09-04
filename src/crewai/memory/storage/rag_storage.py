import logging
import os
import shutil
import uuid
import warnings
from typing import Any, Optional

from chromadb import EmbeddingFunction
from chromadb.api import ClientAPI

from crewai.rag.embeddings.configurator import EmbeddingConfigurator
from crewai.rag.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities.chromadb import create_persistent_client
from crewai.utilities.constants import MAX_FILE_NAME_LENGTH
from crewai.utilities.logger_utils import suppress_logging
from crewai.utilities.paths import db_storage_path


class RAGStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    app: ClientAPI | None = None
    embedder_config: EmbeddingFunction[Any] | None = None

    def __init__(
        self,
        type: str,
        allow_reset: bool = True,
        embedder_config: Any = None,
        crew: Any = None,
        path: Optional[str] = None,
    ) -> None:
        super().__init__(type, allow_reset, embedder_config, crew)
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents
        self.storage_file_name = self._build_storage_file_name(type, agents)

        self.type = type
        self._original_embedder_config = (
            embedder_config  # Store for later use in _set_embedder_config
        )
        self.allow_reset = allow_reset
        self.path = path
        self._initialize_app()

    def _set_embedder_config(self) -> None:
        configurator = EmbeddingConfigurator()
        # Pass the original embedder_config from __init__, not self.embedder_config
        if hasattr(self, "_original_embedder_config"):
            self.embedder_config = configurator.configure_embedder(
                self._original_embedder_config
            )
        else:
            self.embedder_config = configurator.configure_embedder(None)

    def _initialize_app(self) -> None:
        from chromadb.config import Settings

        # Suppress deprecation warnings from chromadb, which are not relevant to us
        # TODO: Remove this once we upgrade chromadb to at least 1.0.8.
        warnings.filterwarnings(
            "ignore",
            message=r".*'model_fields'.*is deprecated.*",
            module=r"^chromadb(\.|$)",
        )

        self._set_embedder_config()

        self.app = create_persistent_client(
            path=self.path if self.path else self.storage_file_name,
            settings=Settings(allow_reset=self.allow_reset),
        )

        self.collection = self.app.get_or_create_collection(
            name=self.type, embedding_function=self.embedder_config
        )
        logging.info(f"Collection found or created: {self.collection}")

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
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()
        try:
            self._generate_embedding(value, metadata)
        except Exception as e:
            logging.error(f"Error during {self.type} save: {str(e)}")

    def search(
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict[str, Any]] = None,
        score_threshold: float = 0.35,
    ) -> list[Any]:
        if not hasattr(self, "app"):
            self._initialize_app()

        try:
            with suppress_logging(
                "chromadb.segment.impl.vector.local_persistent_hnsw", logging.ERROR
            ):
                response = self.collection.query(query_texts=query, n_results=limit)

            results = []
            if (
                response
                and "ids" in response
                and response["ids"]
                and len(response["ids"]) > 0
            ):
                ids_list = (
                    response["ids"][0]
                    if isinstance(response["ids"][0], list)
                    else response["ids"]
                )
                for i in range(len(ids_list)):
                    # Handle metadatas
                    metadata = {}
                    if response.get("metadatas") and len(response["metadatas"]) > 0:
                        metadata_list = (
                            response["metadatas"][0]
                            if isinstance(response["metadatas"][0], list)
                            else response["metadatas"]
                        )
                        if i < len(metadata_list):
                            metadata = metadata_list[i]

                    # Handle documents
                    context = ""
                    if response.get("documents") and len(response["documents"]) > 0:
                        docs_list = (
                            response["documents"][0]
                            if isinstance(response["documents"][0], list)
                            else response["documents"]
                        )
                        if i < len(docs_list):
                            context = docs_list[i]

                    # Handle distances
                    score = 1.0
                    if response.get("distances") and len(response["distances"]) > 0:
                        dist_list = (
                            response["distances"][0]
                            if isinstance(response["distances"][0], list)
                            else response["distances"]
                        )
                        if i < len(dist_list):
                            score = dist_list[i]

                    result = {
                        "id": ids_list[i],
                        "metadata": metadata,
                        "context": context,
                        "score": score,
                    }

                    # Check score threshold - distances are smaller when more similar
                    if isinstance(score, (int, float)) and score <= score_threshold:
                        results.append(result)

            return results
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []

    def _generate_embedding(self, text: str, metadata: dict[str, Any]) -> None:  # type: ignore
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()

        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[str(uuid.uuid4())],
        )

    def reset(self) -> None:
        try:
            if self.app:
                self.app.reset()
                shutil.rmtree(f"{db_storage_path()}/{self.type}")
                self.app = None
                self.collection = None
        except Exception as e:
            if "attempt to write a readonly database" in str(e):
                # Ignore this specific error
                pass
            else:
                raise Exception(
                    f"An error occurred while resetting the {self.type} memory: {e}"
                )

    def _create_default_embedding_function(self) -> EmbeddingFunction[Any]:
        from chromadb.utils.embedding_functions.openai_embedding_function import (
            OpenAIEmbeddingFunction,
        )

        return OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
