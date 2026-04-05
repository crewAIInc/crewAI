"""System prompt generation for A2UI-capable agents."""

from __future__ import annotations

import json

from crewai.a2a.extensions.a2ui.catalog import STANDARD_CATALOG_COMPONENTS
from crewai.a2a.extensions.a2ui.schema import load_schema
from crewai.a2a.extensions.a2ui.server_extension import (
    A2UI_EXTENSION_URI,
    A2UI_V09_BASIC_CATALOG_ID,
)
from crewai.a2a.extensions.a2ui.v0_9 import (
    BASIC_CATALOG_COMPONENTS as V09_CATALOG_COMPONENTS,
    BASIC_CATALOG_FUNCTIONS,
)


def build_a2ui_system_prompt(
    catalog_id: str | None = None,
    allowed_components: list[str] | None = None,
) -> str:
    """Build a v0.8 system prompt fragment instructing the LLM to produce A2UI output.

    Args:
        catalog_id: Catalog identifier to reference. Defaults to the
            standard catalog version derived from ``A2UI_EXTENSION_URI``.
        allowed_components: Subset of component names to expose.  When
            ``None``, all standard catalog components are available.

    Returns:
        A system prompt string to append to the agent's instructions.
    """
    components = sorted(
        allowed_components
        if allowed_components is not None
        else STANDARD_CATALOG_COMPONENTS
    )

    catalog_label = catalog_id or f"standard ({A2UI_EXTENSION_URI.rsplit('/', 1)[-1]})"

    resolved_schema = load_schema(
        "server_to_client_with_standard_catalog", version="v0.8"
    )
    schema_json = json.dumps(resolved_schema, indent=2)

    return f"""\
<A2UI_INSTRUCTIONS>
You can generate rich, declarative UI by emitting A2UI JSON messages.

CATALOG: {catalog_label}
AVAILABLE COMPONENTS: {", ".join(components)}

MESSAGE TYPES (emit exactly ONE per message):
- beginRendering: Initialize a new surface with a root component and optional styles.
- surfaceUpdate: Send/update components for a surface. Each component has a unique id \
and a "component" wrapper containing exactly one component-type key.
- dataModelUpdate: Update the data model for a surface. Data entries have a key and \
one typed value (valueString, valueNumber, valueBoolean, valueMap).
- deleteSurface: Remove a surface.

DATA BINDING:
- Use {{"literalString": "..."}} for inline string values.
- Use {{"literalNumber": ...}} for inline numeric values.
- Use {{"literalBoolean": ...}} for inline boolean values.
- Use {{"literalArray": ["...", "..."]}} for inline array values.
- Use {{"path": "/data/model/path"}} to bind to data model values.

ACTIONS:
- Interactive components (Button, etc.) have an "action" with a "name" and optional \
"context" array of key/value pairs.
- Values in action context can use data binding (path or literal).

OUTPUT FORMAT:
Emit each A2UI message as a valid JSON object. When generating UI, produce a \
beginRendering message first, then surfaceUpdate messages with components, and \
optionally dataModelUpdate messages to populate data-bound values.

SCHEMA:
{schema_json}
</A2UI_INSTRUCTIONS>"""


def build_a2ui_v09_system_prompt(
    catalog_id: str | None = None,
    allowed_components: list[str] | None = None,
) -> str:
    """Build a v0.9 system prompt fragment instructing the LLM to produce A2UI output.

    Args:
        catalog_id: Catalog identifier to reference. Defaults to the
            v0.9 basic catalog.
        allowed_components: Subset of component names to expose.  When
            ``None``, all basic catalog components are available.

    Returns:
        A system prompt string to append to the agent's instructions.
    """
    components = sorted(
        allowed_components if allowed_components is not None else V09_CATALOG_COMPONENTS
    )

    catalog_label = catalog_id or A2UI_V09_BASIC_CATALOG_ID
    functions = sorted(BASIC_CATALOG_FUNCTIONS)

    envelope_schema = load_schema("server_to_client", version="v0.9")
    schema_json = json.dumps(envelope_schema, indent=2)

    return f"""\
<A2UI_INSTRUCTIONS>
You can generate rich, declarative UI by emitting A2UI v0.9 JSON messages.
Every message MUST include "version": "v0.9".

CATALOG: {catalog_label}
AVAILABLE COMPONENTS: {", ".join(components)}
AVAILABLE FUNCTIONS: {", ".join(functions)}

MESSAGE TYPES (emit exactly ONE per message alongside "version": "v0.9"):
- createSurface: Create a new surface. Requires surfaceId and catalogId. \
Optionally includes theme (primaryColor, iconUrl, agentDisplayName) and \
sendDataModel (boolean).
- updateComponents: Send/update components for a surface. Each component is a flat \
object with "id", "component" (type name string), and type-specific properties at the \
top level. One component MUST have id "root".
- updateDataModel: Update the data model. Uses "path" (JSON Pointer) and "value" \
(any JSON type). Omit "value" to delete the key at path.
- deleteSurface: Remove a surface by surfaceId.

COMPONENT FORMAT (flat, NOT nested):
{{"id": "myText", "component": "Text", "text": "Hello world", "variant": "h1"}}
{{"id": "myBtn", "component": "Button", "child": "myText", "action": {{"event": \
{{"name": "click"}}}}}}

DATA BINDING:
- Use plain values for literals: "text": "Hello" or "value": 42
- Use {{"path": "/data/model/path"}} to bind to data model values.
- Use {{"call": "functionName", "args": {{...}}}} for client-side functions.

ACTIONS:
- Server event: {{"event": {{"name": "actionName", "context": {{"key": "value"}}}}}}
- Local function: {{"functionCall": {{"call": "openUrl", "args": {{"url": "..."}}}}}}

OUTPUT FORMAT:
Emit each A2UI message as a valid JSON object. When generating UI, first emit a \
createSurface message with the catalogId, then updateComponents messages with \
components (one must have id "root"), and optionally updateDataModel messages.

ENVELOPE SCHEMA:
{schema_json}
</A2UI_INSTRUCTIONS>"""
