"""A2UI server extension for the A2A protocol."""

from __future__ import annotations

import logging
from typing import Any

from crewai.a2a.extensions.a2ui.models import A2UIResponse, extract_a2ui_json_objects
from crewai.a2a.extensions.a2ui.v0_9 import (
    extract_a2ui_v09_json_objects,
)
from crewai.a2a.extensions.a2ui.validator import (
    A2UIValidationError,
    validate_a2ui_message,
    validate_a2ui_message_v09,
)
from crewai.a2a.extensions.server import ExtensionContext, ServerExtension


logger = logging.getLogger(__name__)

A2UI_MIME_TYPE = "application/json+a2ui"
A2UI_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.8"
A2UI_STANDARD_CATALOG_ID = (
    "https://a2ui.org/specification/v0_8/standard_catalog_definition.json"
)
A2UI_V09_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.9"
A2UI_V09_BASIC_CATALOG_ID = "https://a2ui.org/specification/v0_9/basic_catalog.json"


class A2UIServerExtension(ServerExtension):
    """A2A server extension that enables A2UI declarative UI generation.

    Supports both v0.8 and v0.9 of the A2UI protocol via the ``version``
    parameter.  When activated by a client, this extension:

    * Negotiates catalog preferences during ``on_request``.
    * Wraps A2UI messages in the agent response as A2A DataParts with
      ``application/json+a2ui`` MIME type during ``on_response``.

    Example::

        A2AServerConfig
            server_extensions=[A2UIServerExtension],
            default_output_modes=["text/plain", "application/json+a2ui"],
    """

    uri: str = A2UI_EXTENSION_URI
    required: bool = False
    description: str = "A2UI declarative UI generation"

    def __init__(
        self,
        catalog_ids: list[str] | None = None,
        accept_inline_catalogs: bool = False,
        version: str = "v0.8",
    ) -> None:
        """Initialize the A2UI server extension.

        Args:
            catalog_ids: Catalog identifiers this server supports.
            accept_inline_catalogs: Whether inline catalog definitions are accepted.
            version: Protocol version, ``"v0.8"`` or ``"v0.9"``.
        """
        self._catalog_ids = catalog_ids or []
        self._accept_inline_catalogs = accept_inline_catalogs
        self._version = version
        if version == "v0.9":
            self.uri = A2UI_V09_EXTENSION_URI

    @property
    def params(self) -> dict[str, Any]:
        """Extension parameters advertised in the AgentCard."""
        result: dict[str, Any] = {}
        if self._catalog_ids:
            result["supportedCatalogIds"] = self._catalog_ids
        result["acceptsInlineCatalogs"] = self._accept_inline_catalogs
        return result

    async def on_request(self, context: ExtensionContext) -> None:
        """Extract A2UI catalog preferences from the client request.

        Stores the negotiated catalog in ``context.state`` under
        ``"a2ui_catalog_id"`` for downstream use.
        """
        if not self.is_active(context):
            return

        catalog_id = context.get_extension_metadata(self.uri, "catalogId")
        if isinstance(catalog_id, str):
            context.state["a2ui_catalog_id"] = catalog_id
        elif self._catalog_ids:
            context.state["a2ui_catalog_id"] = self._catalog_ids[0]

        context.state["a2ui_active"] = True

    async def on_response(self, context: ExtensionContext, result: Any) -> Any:
        """Wrap A2UI messages in the result as A2A DataParts.

        Scans the result for A2UI JSON payloads and converts them into
        DataParts with ``application/json+a2ui`` MIME type and A2UI metadata.
        Dispatches to the correct extractor and validator based on version.
        """
        if not context.state.get("a2ui_active"):
            return result

        if not isinstance(result, str):
            return result

        if self._version == "v0.9":
            a2ui_messages = extract_a2ui_v09_json_objects(result)
        else:
            a2ui_messages = extract_a2ui_json_objects(result)

        if not a2ui_messages:
            return result

        build_fn = _build_data_part_v09 if self._version == "v0.9" else _build_data_part
        data_parts = [
            part
            for part in (build_fn(msg_data) for msg_data in a2ui_messages)
            if part is not None
        ]

        if not data_parts:
            return result

        return A2UIResponse(text=result, a2ui_parts=data_parts)


def _build_data_part(msg_data: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a v0.8 A2UI message and wrap it as a DataPart dict."""
    try:
        validated = validate_a2ui_message(msg_data)
    except A2UIValidationError:
        logger.warning("Skipping invalid A2UI message in response", exc_info=True)
        return None
    return {
        "kind": "data",
        "data": validated.model_dump(by_alias=True, exclude_none=True),
        "metadata": {
            "mimeType": A2UI_MIME_TYPE,
        },
    }


def _build_data_part_v09(msg_data: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a v0.9 A2UI message and wrap it as a DataPart dict."""
    try:
        validated = validate_a2ui_message_v09(msg_data)
    except A2UIValidationError:
        logger.warning("Skipping invalid A2UI v0.9 message in response", exc_info=True)
        return None
    return {
        "kind": "data",
        "data": validated.model_dump(by_alias=True, exclude_none=True),
        "metadata": {
            "mimeType": A2UI_MIME_TYPE,
        },
    }
