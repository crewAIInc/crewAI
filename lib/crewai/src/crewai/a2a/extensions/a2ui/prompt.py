"""System prompt generation for A2UI-capable agents."""

from __future__ import annotations

import json

from crewai.a2a.extensions.a2ui.catalog import STANDARD_CATALOG_COMPONENTS
from crewai.a2a.extensions.a2ui.schema import load_schema


def build_a2ui_system_prompt(
    catalog_id: str | None = None,
    allowed_components: list[str] | None = None,
) -> str:
    """Build a system prompt fragment instructing the LLM to produce A2UI output.

    The prompt describes the A2UI message format, available components, and
    data binding rules.  It includes the resolved schema so the LLM can
    generate structured output.

    Args:
        catalog_id: Catalog identifier to reference. Defaults to the
            standard catalog for A2UI v0.8.
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

    catalog_label = catalog_id or "standard (v0.8)"

    resolved_schema = load_schema("server_to_client_with_standard_catalog")
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
