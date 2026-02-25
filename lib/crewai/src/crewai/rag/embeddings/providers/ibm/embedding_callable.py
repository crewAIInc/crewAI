"""IBM WatsonX embedding function implementation."""

from typing import Any, cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.ibm.types import WatsonXProviderConfig
from crewai.utilities.printer import Printer


_printer = Printer()


class WatsonXEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for IBM WatsonX models."""

    def __init__(
        self, *, verbose: bool = True, **kwargs: Unpack[WatsonXProviderConfig]
    ) -> None:
        """Initialize WatsonX embedding function.

        Args:
            verbose: Whether to print error messages.
            **kwargs: Configuration parameters for WatsonX Embeddings and Credentials.
        """
        super().__init__(**kwargs)
        self._config = kwargs
        self._verbose = verbose

    @staticmethod
    def name() -> str:
        """Return the name of the embedding function for ChromaDB compatibility."""
        return "watsonx"

    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for input documents.

        Args:
            input: List of documents to embed.

        Returns:
            List of embedding vectors.
        """
        try:
            from ibm_watsonx_ai import (  # type: ignore[import-untyped]
                Credentials,
            )
            import ibm_watsonx_ai.foundation_models as watson_models  # type: ignore[import-untyped]
            from ibm_watsonx_ai.metanames import (  # type: ignore[import-untyped]
                EmbedTextParamsMetaNames as EmbedParams,
            )

        except ImportError as e:
            raise ImportError(
                "ibm-watsonx-ai is required for watsonx embeddings. "
                "Install it with: uv add ibm-watsonx-ai"
            ) from e

        if isinstance(input, str):
            input = [input]

        embeddings_config: dict[str, Any] = {
            "model_id": self._config["model_id"],
        }
        if "params" in self._config and self._config["params"] is not None:
            embeddings_config["params"] = self._config["params"]
        if "project_id" in self._config and self._config["project_id"] is not None:
            embeddings_config["project_id"] = self._config["project_id"]
        if "space_id" in self._config and self._config["space_id"] is not None:
            embeddings_config["space_id"] = self._config["space_id"]
        if "api_client" in self._config and self._config["api_client"] is not None:
            embeddings_config["api_client"] = self._config["api_client"]
        if "verify" in self._config and self._config["verify"] is not None:
            embeddings_config["verify"] = self._config["verify"]
        if "persistent_connection" in self._config:
            embeddings_config["persistent_connection"] = self._config[
                "persistent_connection"
            ]
        if "batch_size" in self._config:
            embeddings_config["batch_size"] = self._config["batch_size"]
        if "concurrency_limit" in self._config:
            embeddings_config["concurrency_limit"] = self._config["concurrency_limit"]
        if "max_retries" in self._config and self._config["max_retries"] is not None:
            embeddings_config["max_retries"] = self._config["max_retries"]
        if "delay_time" in self._config and self._config["delay_time"] is not None:
            embeddings_config["delay_time"] = self._config["delay_time"]
        if (
            "retry_status_codes" in self._config
            and self._config["retry_status_codes"] is not None
        ):
            embeddings_config["retry_status_codes"] = self._config["retry_status_codes"]

        if "credentials" in self._config and self._config["credentials"] is not None:
            embeddings_config["credentials"] = self._config["credentials"]
        else:
            cred_config: dict[str, Any] = {}
            if "url" in self._config and self._config["url"] is not None:
                cred_config["url"] = self._config["url"]
            if "api_key" in self._config and self._config["api_key"] is not None:
                cred_config["api_key"] = self._config["api_key"]
            if "name" in self._config and self._config["name"] is not None:
                cred_config["name"] = self._config["name"]
            if (
                "iam_serviceid_crn" in self._config
                and self._config["iam_serviceid_crn"] is not None
            ):
                cred_config["iam_serviceid_crn"] = self._config["iam_serviceid_crn"]
            if (
                "trusted_profile_id" in self._config
                and self._config["trusted_profile_id"] is not None
            ):
                cred_config["trusted_profile_id"] = self._config["trusted_profile_id"]
            if "token" in self._config and self._config["token"] is not None:
                cred_config["token"] = self._config["token"]
            if (
                "projects_token" in self._config
                and self._config["projects_token"] is not None
            ):
                cred_config["projects_token"] = self._config["projects_token"]
            if "username" in self._config and self._config["username"] is not None:
                cred_config["username"] = self._config["username"]
            if "password" in self._config and self._config["password"] is not None:
                cred_config["password"] = self._config["password"]
            if (
                "instance_id" in self._config
                and self._config["instance_id"] is not None
            ):
                cred_config["instance_id"] = self._config["instance_id"]
            if "version" in self._config and self._config["version"] is not None:
                cred_config["version"] = self._config["version"]
            if (
                "bedrock_url" in self._config
                and self._config["bedrock_url"] is not None
            ):
                cred_config["bedrock_url"] = self._config["bedrock_url"]
            if (
                "platform_url" in self._config
                and self._config["platform_url"] is not None
            ):
                cred_config["platform_url"] = self._config["platform_url"]
            if "proxies" in self._config and self._config["proxies"] is not None:
                cred_config["proxies"] = self._config["proxies"]
            if (
                "verify" not in embeddings_config
                and "verify" in self._config
                and self._config["verify"] is not None
            ):
                cred_config["verify"] = self._config["verify"]

            if cred_config:
                embeddings_config["credentials"] = Credentials(**cred_config)

        if "params" not in embeddings_config:
            embeddings_config["params"] = {
                EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
                EmbedParams.RETURN_OPTIONS: {"input_text": True},
            }

        embedding = watson_models.Embeddings(**embeddings_config)

        try:
            embeddings = embedding.embed_documents(input)
            return cast(Embeddings, embeddings)
        except Exception as e:
            if self._verbose:
                _printer.print(f"Error during WatsonX embedding: {e}", color="red")
            raise
