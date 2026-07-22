"""True async HTTP client for OCI Generative AI.

Bypasses the synchronous OCI SDK for HTTP, using aiohttp directly
with OCI request signing. This gives real async I/O instead of
thread-pool wrappers, until the OCI SDK ships native async support.
"""

from __future__ import annotations

import json
import ssl
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, AsyncIterator

import aiohttp
import certifi
import requests


def _get_oci_genai_api_version() -> str:
    """Detect OCI GenAI API version from SDK, fallback to known version."""
    try:
        from oci.generative_ai_inference import GenerativeAiInferenceClient

        if hasattr(GenerativeAiInferenceClient, "API_VERSION"):
            return GenerativeAiInferenceClient.API_VERSION
    except ImportError:
        pass
    return "20231130"


OCI_GENAI_API_VERSION = _get_oci_genai_api_version()


def _snake_to_camel(name: str) -> str:
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _convert_keys_to_camel(obj: Any) -> Any:
    """Recursively convert dict keys from snake_case to camelCase."""
    if isinstance(obj, dict):
        return {_snake_to_camel(k): _convert_keys_to_camel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_keys_to_camel(item) for item in obj]
    return obj


class OCIAsyncClient:
    """Async HTTP client for OCI Generative AI services.

    Uses aiohttp with OCI request signing for true async I/O.
    Reuses aiohttp.ClientSession for connection pooling.
    """

    def __init__(
        self,
        service_endpoint: str,
        signer: Any,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.service_endpoint = service_endpoint.rstrip("/")
        self.signer = signer
        self.config = config or {}
        self._session: aiohttp.ClientSession | None = None
        self._ensure_signer()

    def _ensure_signer(self) -> None:
        if self.signer is not None:
            return
        if self.config:
            try:
                from oci.signer import Signer

                self.signer = Signer.from_config(self.config)
            except Exception as e:
                raise ValueError(f"Failed to create OCI signer from config: {e}") from e

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> OCIAsyncClient:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    def _sign_headers(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> dict[str, str]:
        req = requests.Request(method, url, json=body)
        prepared = req.prepare()
        signed = self.signer(prepared)
        headers = dict(signed.headers)
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    @asynccontextmanager
    async def _arequest(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        json_body: dict[str, Any] | None = None,
        timeout: int = 300,
    ) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        session = await self._get_session()
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with session.request(
            method, url, headers=headers, json=json_body, timeout=client_timeout
        ) as response:
            yield response

    async def _parse_sse_async(
        self, content: aiohttp.StreamReader
    ) -> AsyncIterator[dict[str, Any]]:
        async for line in content:
            line = line.strip()
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.lower().startswith("data:"):
                data = decoded[5:].strip()
                if data and not data.startswith("[DONE]"):
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        continue

    async def chat_async(
        self,
        compartment_id: str,
        chat_request_dict: dict[str, Any],
        serving_mode_dict: dict[str, Any],
        stream: bool = False,
        timeout: int = 300,
    ) -> AsyncIterator[dict[str, Any]]:
        """Make async chat request to OCI GenAI.

        Yields SSE events for streaming, or a single response dict.
        """
        url = f"{self.service_endpoint}/{OCI_GENAI_API_VERSION}/actions/chat"

        body = {
            "compartmentId": compartment_id,
            "servingMode": _convert_keys_to_camel(serving_mode_dict),
            "chatRequest": _convert_keys_to_camel(chat_request_dict),
        }

        headers = self._sign_headers("POST", url, body, stream=stream)

        async with self._arequest("POST", url, headers, body, timeout) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"OCI GenAI async request failed ({response.status}): {error_text}"
                )

            if stream:
                async for event in self._parse_sse_async(response.content):
                    yield event
            else:
                data = await response.json()
                yield data
