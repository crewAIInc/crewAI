"""OCI embedding function implementation."""

from __future__ import annotations

import base64
from collections.abc import Iterator, Sequence
import mimetypes
import os
from pathlib import Path
from typing import Any, cast

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from typing_extensions import Unpack

from crewai.rag.embeddings.providers.oci.types import OCIProviderConfig
from crewai.utilities.oci import create_oci_client_kwargs, get_oci_module


CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"
DEFAULT_OCI_REGION = "us-chicago-1"


def _get_oci_module() -> Any:
    """Backward-compatible module-local alias used by tests and patches."""
    return get_oci_module()


class OCIEmbeddingFunction(EmbeddingFunction[Documents]):
    """Embedding function for OCI Generative AI embedding models."""

    def __init__(self, **kwargs: Unpack[OCIProviderConfig]) -> None:
        self._config = kwargs
        self._client: Any = kwargs.get("client")
        if self._client is None:
            service_endpoint = kwargs.get("service_endpoint")
            region = kwargs.get("region") or os.getenv("OCI_REGION", DEFAULT_OCI_REGION)
            if service_endpoint is None:
                service_endpoint = (
                    f"https://inference.generativeai.{region}.oci.oraclecloud.com"
                )

            client_kwargs = create_oci_client_kwargs(
                auth_type=kwargs.get("auth_type", "API_KEY"),
                service_endpoint=service_endpoint,
                auth_file_location=kwargs.get("auth_file_location", "~/.oci/config"),
                auth_profile=kwargs.get("auth_profile", "DEFAULT"),
                timeout=kwargs.get("timeout", (10, 120)),
                oci_module=_get_oci_module(),
            )
            self._client = (
                _get_oci_module().generative_ai_inference.GenerativeAiInferenceClient(
                    **client_kwargs
                )
            )

    def _require_client(self) -> Any:
        if self._client is None:
            raise ValueError("OCI embedding client is not initialized.")
        return self._client

    @staticmethod
    def name() -> str:
        return "oci"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> OCIEmbeddingFunction:
        timeout = config.get("timeout")
        if isinstance(timeout, list):
            config = dict(config)
            config["timeout"] = tuple(timeout)
        return OCIEmbeddingFunction(**config)

    def get_config(self) -> dict[str, Any]:
        config = dict(self._config)
        config.pop("client", None)
        timeout = config.get("timeout")
        if isinstance(timeout, tuple):
            config["timeout"] = list(timeout)
        return config

    def _get_serving_mode(self) -> Any:
        oci = _get_oci_module()
        model_name = self._config.get("model_name")
        if not model_name:
            raise ValueError("OCI embeddings require model_name")
        if model_name.startswith(CUSTOM_ENDPOINT_PREFIX):
            return oci.generative_ai_inference.models.DedicatedServingMode(
                endpoint_id=model_name
            )
        return oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=model_name
        )

    def _build_request(
        self, inputs: list[str], *, input_type: str | None = None
    ) -> Any:
        oci = _get_oci_module()
        compartment_id = self._config.get("compartment_id") or os.getenv(
            "OCI_COMPARTMENT_ID"
        )
        if not compartment_id:
            raise ValueError(
                "OCI embeddings require compartment_id. Set it explicitly or use OCI_COMPARTMENT_ID."
            )

        request_kwargs: dict[str, Any] = {
            "serving_mode": self._get_serving_mode(),
            "compartment_id": compartment_id,
            "truncate": self._config.get("truncate", "END"),
            "inputs": inputs,
        }

        resolved_input_type = input_type or self._config.get("input_type")
        if resolved_input_type:
            request_kwargs["input_type"] = resolved_input_type

        output_dimensions = self._config.get("output_dimensions")
        if output_dimensions is not None:
            embed_text_details = oci.generative_ai_inference.models.EmbedTextDetails
            if hasattr(embed_text_details, "output_dimensions"):
                request_kwargs["output_dimensions"] = output_dimensions
            else:
                raise ValueError(
                    "output_dimensions requires a newer OCI SDK. Upgrade the oci package."
                )

        return oci.generative_ai_inference.models.EmbedTextDetails(**request_kwargs)

    def _batch_inputs(self, input: list[str]) -> Iterator[list[str]]:
        batch_size = self._config.get("batch_size", 96)
        for index in range(0, len(input), batch_size):
            yield input[index : index + batch_size]

    @staticmethod
    def _to_data_uri(image: str | bytes | Path, mime_type: str = "image/png") -> str:
        if isinstance(image, Path):
            resolved_mime = mimetypes.guess_type(image.name)[0] or mime_type
            data = image.read_bytes()
            return (
                f"data:{resolved_mime};base64,"
                f"{base64.b64encode(data).decode('ascii')}"
            )
        if isinstance(image, bytes):
            return f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
        if image.startswith("data:"):
            return image
        path = Path(image)
        if path.exists():
            return OCIEmbeddingFunction._to_data_uri(path, mime_type=mime_type)
        raise ValueError(
            "OCI image embeddings require a file path, raw bytes, or a data URI."
        )

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings: Embeddings = []
        for chunk in self._batch_inputs(input):
            response = self._require_client().embed_text(self._build_request(chunk))
            embeddings.extend(cast(Embeddings, response.data.embeddings))
        return embeddings

    def embed_image(
        self, image: str | bytes | Path, *, mime_type: str = "image/png"
    ) -> list[float]:
        return [
            float(value)
            for value in self.embed_image_batch([image], mime_type=mime_type)[0]
        ]

    def embed_image_batch(
        self, images: Sequence[str | bytes | Path], *, mime_type: str = "image/png"
    ) -> Embeddings:
        embeddings: Embeddings = []
        for image in images:
            data_uri = self._to_data_uri(image, mime_type=mime_type)
            response = self._require_client().embed_text(
                self._build_request([data_uri], input_type="IMAGE")
            )
            embeddings.extend(cast(Embeddings, response.data.embeddings))
        return embeddings
