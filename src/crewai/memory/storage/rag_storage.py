import contextlib
import io
import logging
import os
import shutil
import uuid
from typing import Any, Dict, List, Optional
from crewai.memory.storage.base_rag_storage import BaseRAGStorage
from crewai.utilities.paths import db_storage_path
from chromadb.api import ClientAPI
from chromadb.api.types import validate_embedding_function
from chromadb import Documents, EmbeddingFunction, Embeddings
from typing import cast


@contextlib.contextmanager
def suppress_logging(
    logger_name="chromadb.segment.impl.vector.local_persistent_hnsw",
    level=logging.ERROR,
):
    logger = logging.getLogger(logger_name)
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ), contextlib.suppress(UserWarning):
        yield
    logger.setLevel(original_level)


class RAGStorage(BaseRAGStorage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency.
    """

    app: ClientAPI | None = None

    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        super().__init__(type, allow_reset, embedder_config, crew)
        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents

        self.type = type

        self.allow_reset = allow_reset
        self._initialize_app()

    def _set_embedder_config(self):
        import chromadb.utils.embedding_functions as embedding_functions

        if self.embedder_config is None:
            self.embedder_config = self._create_default_embedding_function()

        if isinstance(self.embedder_config, dict):
            provider = self.embedder_config.get("provider")
            config = self.embedder_config.get("config", {})
            model_name = config.get("model")
            if provider == "openai":
                self.embedder_config = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                    model_name=model_name,
                )
            elif provider == "azure":
                self.embedder_config = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=config.get("api_key"),
                    api_base=config.get("api_base"),
                    api_type=config.get("api_type", "azure"),
                    api_version=config.get("api_version"),
                    model_name=model_name,
                )
            elif provider == "ollama":
                from openai import OpenAI

                class OllamaEmbeddingFunction(EmbeddingFunction):
                    def __call__(self, input: Documents) -> Embeddings:
                        client = OpenAI(
                            base_url="http://localhost:11434/v1",
                            api_key=config.get("api_key", "ollama"),
                        )
                        try:
                            response = client.embeddings.create(
                                input=input, model=model_name
                            )
                            embeddings = [item.embedding for item in response.data]
                            return cast(Embeddings, embeddings)
                        except Exception as e:
                            raise e

                self.embedder_config = OllamaEmbeddingFunction()
            elif provider == "vertexai":
                self.embedder_config = (
                    embedding_functions.GoogleVertexEmbeddingFunction(
                        model_name=model_name,
                        api_key=config.get("api_key"),
                    )
                )
            elif provider == "google":
                self.embedder_config = (
                    embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                        model_name=model_name,
                        api_key=config.get("api_key"),
                    )
                )
            elif provider == "cohere":
                self.embedder_config = embedding_functions.CohereEmbeddingFunction(
                    model_name=model_name,
                    api_key=config.get("api_key"),
                )
            elif provider == "huggingface":
                self.embedder_config = embedding_functions.HuggingFaceEmbeddingServer(
                    url=config.get("api_url"),
                )
            else:
                raise Exception(
                    f"Unsupported embedding provider: {provider}, supported providers: [openai, azure, ollama, vertexai, google, cohere, huggingface]"
                )
        else:
            validate_embedding_function(self.embedder_config)  # type: ignore # used for validating embedder_config if defined a embedding function/class
            self.embedder_config = self.embedder_config

    def _initialize_app(self):
        import chromadb
        from chromadb.config import Settings

        self._set_embedder_config()
        chroma_client = chromadb.PersistentClient(
            path=f"{db_storage_path()}/{self.type}/{self.agents}",
            settings=Settings(allow_reset=self.allow_reset),
        )

        self.app = chroma_client

        try:
            self.collection = self.app.get_collection(
                name=self.type, embedding_function=self.embedder_config
            )
        except Exception:
            self.collection = self.app.create_collection(
                name=self.type, embedding_function=self.embedder_config
            )

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
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
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Any]:
        if not hasattr(self, "app"):
            self._initialize_app()

        try:
            with suppress_logging():
                response = self.collection.query(query_texts=query, n_results=limit)

            results = []
            for i in range(len(response["ids"][0])):
                result = {
                    "id": response["ids"][0][i],
                    "metadata": response["metadatas"][0][i],
                    "context": response["documents"][0][i],
                    "score": response["distances"][0][i],
                }
                if result["score"] >= score_threshold:
                    results.append(result)

            return results
        except Exception as e:
            logging.error(f"Error during {self.type} search: {str(e)}")
            return []

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> None:  # type: ignore
        if not hasattr(self, "app") or not hasattr(self, "collection"):
            self._initialize_app()

        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[str(uuid.uuid4())],
        )

    def reset(self) -> None:
        try:
            shutil.rmtree(f"{db_storage_path()}/{self.type}")
            if self.app:
                self.app.reset()
        except Exception as e:
            if "attempt to write a readonly database" in str(e):
                # Ignore this specific error
                pass
            else:
                raise Exception(
                    f"An error occurred while resetting the {self.type} memory: {e}"
                )

    def _create_default_embedding_function(self):
        import chromadb.utils.embedding_functions as embedding_functions

        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
