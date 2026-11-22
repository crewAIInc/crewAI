"""Public Ace model discovery shared by the CLI and native LLM factory."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time

import httpx


CATALOG_URL = "https://platform.acedata.cloud/api/v1/models/catalog/"
API_BASE = "https://api.acedata.cloud/v1"
_ENDPOINTS = {
    "chat_completions": ("/v1/chat/completions", API_BASE),
    "messages": ("/v1/messages", "https://api.acedata.cloud"),
    "responses": ("/v1/responses", API_BASE),
}


@dataclass(frozen=True)
class AceModel:
    """Only explicit metadata is used; limits and prices are not inferred."""

    id: str
    label: str
    protocols: tuple[str, ...]
    vision: bool


@lru_cache(maxsize=1)
def _catalog(bucket: int) -> tuple[AceModel, ...]:
    response = httpx.get(CATALOG_URL, timeout=6, follow_redirects=False)
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("items"), list):
        raise ValueError("Invalid Ace Data Cloud catalog")
    models = {}
    for row in body["items"]:
        if not isinstance(row, dict) or row.get("modality") != "chat":
            continue
        model_id = row.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            continue
        endpoints = row.get("endpoints")
        if not isinstance(endpoints, list):
            raise ValueError("Ace Data Cloud catalog lacks protocol metadata")
        protocols = tuple(
            protocol
            for protocol, (path, base) in _ENDPOINTS.items()
            if any(
                isinstance(endpoint, dict)
                and endpoint.get("protocol") == protocol
                and endpoint.get("path") == path
                and endpoint.get("base_url") == base
                and endpoint.get("auth") == "bearer"
                for endpoint in endpoints
            )
        )
        if not protocols:
            raise ValueError(f"No supported published protocol for {model_id}")
        capabilities = row.get("capabilities")
        label = row.get("name")
        models[model_id] = AceModel(
            model_id,
            label if isinstance(label, str) else model_id,
            protocols,
            isinstance(capabilities, list) and "vision" in capabilities,
        )
    if not models:
        raise ValueError("Ace Data Cloud returned no chat models")
    return tuple(models.values())


def get_ace_models() -> tuple[AceModel, ...]:
    """Return the complete typed catalog, cached for at most five minutes."""
    return _catalog(int(time.monotonic() // 300))
