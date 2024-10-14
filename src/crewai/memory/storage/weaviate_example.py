import contextlib
import io
import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from crewai.memory.storage.interface import Storage
from crewai.utilities.paths import db_storage_path
import weaviate
import weaviate.exceptions as weaviate_exceptions
import weaviate.classes as wvc


@contextlib.contextmanager
def suppress_logging(
    logger_name="weaviate",
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


class WeaviateStorage(Storage):
    """
    Extends Storage to handle embeddings for memory entries, improving
    search efficiency using Weaviate.
    """

    def __init__(self, type, allow_reset=True, embedder_config=None, crew=None):
        super().__init__()
        if (
            not os.getenv("OPENAI_API_KEY")
            and not os.getenv("OPENAI_BASE_URL") == "https://api.openai.com/v1"
        ):
            os.environ["OPENAI_API_KEY"] = "fake"

        agents = crew.agents if crew else []
        agents = [self._sanitize_role(agent.role) for agent in agents]
        agents = "_".join(agents)
        self.agents = agents

        # config = {

        # }

        # if embedder_config:
        #     config["embedder"] = embedder_config
        self.type = type
        # self.config = config
        self.allow_reset = allow_reset
        self.app: Any = None  # Initialize app attribute
        self._initialize_app()  # Call _initialize_app in the constructor

    def _initialize_app(self):
        if self.app is None:
            try:
                self.app = weaviate.connect_to_embedded(
                    hostname="localhost",
                    port=8079,
                    grpc_port=50050,
                    persistence_data_path=f"{db_storage_path()}/{self.type}/{self.agents}",
                    headers={
                        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY"),
                    },
                )
                print("self.app", self.app)
            except weaviate_exceptions.WeaviateStartUpError:
                # If the embedded instance is already running, connect to it
                self.app = weaviate.connect_to_local(
                    port=8079,
                    grpc_port=50050,
                    headers={
                        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY"),
                    },
                )
        self.app.collections.delete(self.type)
        # generate collection based of type: stm, entity
        self.app.collections.create(
            name=self.type,
            description="Collection for storing memory items",
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),  # If set to "none" you must always provide vectors yourself. Could be any other "text2vec-*" also.
        )

    def _sanitize_role(self, role: str) -> str:
        """
        Sanitizes agent roles to ensure valid directory names.
        """
        return role.replace("\n", "").replace(" ", "_").replace("/", "_")

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        if not hasattr(self, "app"):
            self._initialize_app()
        self._generate_embedding(value, metadata)

    def search(  # type: ignore # BUG?: Signature of "search" incompatible with supertype "Storage"
        self,
        query: str,
        limit: int = 3,
        filter: Optional[dict] = None,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        if not hasattr(self, "app"):
            self._initialize_app()

        collection = self.app.collections.get(self.type)

        response = collection.query.near_text(
            query,
            limit=limit,
            distance=score_threshold,
        )

        results = []
        for obj in response.objects:
            if obj.score >= score_threshold:
                result = {
                    "metadata": obj.properties.get("metadata", {}),
                    "context": obj.properties.get("text", ""),
                    "score": obj.score,
                }
                results.append(result)

        return results

    def _generate_embedding(self, text: str, metadata: Dict[str, Any]) -> Any:
        if not hasattr(self, "app"):
            self._initialize_app()
        collection = self.app.collections.get(self.type)
        collection.data.insert({"text": text, "metadata": metadata})

    def reset(self) -> None:
        try:
            shutil.rmtree(f"{db_storage_path()}/{self.type}")
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the {self.type} memory: {e}"
            )
