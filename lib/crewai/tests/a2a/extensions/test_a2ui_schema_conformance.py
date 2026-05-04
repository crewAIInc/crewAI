"""Cross-validate A2UI Pydantic models against vendored JSON schemas.

Ensures the two validation sources stay in sync: representative payloads
must be accepted or rejected consistently by both the Pydantic models and
the JSON schemas.
"""

from __future__ import annotations

from typing import Any

import jsonschema
import pytest

from crewai.a2a.extensions.a2ui import catalog
from crewai.a2a.extensions.a2ui.models import A2UIEvent, A2UIMessage
from crewai.a2a.extensions.a2ui.schema import load_schema


SERVER_SCHEMA = load_schema("server_to_client")
CLIENT_SCHEMA = load_schema("client_to_server")
CATALOG_SCHEMA = load_schema("standard_catalog_definition")


def _json_schema_valid(schema: dict[str, Any], instance: dict[str, Any]) -> bool:
    """Return True if *instance* validates against *schema*."""
    try:
        jsonschema.validate(instance, schema)
        return True
    except jsonschema.ValidationError:
        return False


def _pydantic_valid_message(data: dict[str, Any]) -> bool:
    """Return True if *data* validates as an A2UIMessage."""
    try:
        A2UIMessage.model_validate(data)
        return True
    except Exception:
        return False


def _pydantic_valid_event(data: dict[str, Any]) -> bool:
    """Return True if *data* validates as an A2UIEvent."""
    try:
        A2UIEvent.model_validate(data)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Valid server-to-client payloads
# ---------------------------------------------------------------------------

VALID_SERVER_MESSAGES: list[dict[str, Any]] = [
    {
        "beginRendering": {
            "surfaceId": "s1",
            "root": "root-col",
        },
    },
    {
        "beginRendering": {
            "surfaceId": "s2",
            "root": "root-col",
            "catalogId": "standard (v0.8)",
            "styles": {"primaryColor": "#FF0000", "font": "Roboto"},
        },
    },
    {
        "surfaceUpdate": {
            "surfaceId": "s1",
            "components": [
                {
                    "id": "title",
                    "component": {
                        "Text": {"text": {"literalString": "Hello"}},
                    },
                },
            ],
        },
    },
    {
        "surfaceUpdate": {
            "surfaceId": "s1",
            "components": [
                {
                    "id": "weighted",
                    "weight": 2.0,
                    "component": {
                        "Column": {
                            "children": {"explicitList": ["a", "b"]},
                        },
                    },
                },
            ],
        },
    },
    {
        "dataModelUpdate": {
            "surfaceId": "s1",
            "contents": [
                {"key": "name", "valueString": "Alice"},
                {"key": "score", "valueNumber": 42},
                {"key": "active", "valueBoolean": True},
            ],
        },
    },
    {
        "dataModelUpdate": {
            "surfaceId": "s1",
            "path": "/user",
            "contents": [
                {
                    "key": "prefs",
                    "valueMap": [
                        {"key": "theme", "valueString": "dark"},
                    ],
                },
            ],
        },
    },
    {
        "deleteSurface": {"surfaceId": "s1"},
    },
]

# ---------------------------------------------------------------------------
# Invalid server-to-client payloads
# ---------------------------------------------------------------------------

INVALID_SERVER_MESSAGES: list[dict[str, Any]] = [
    {},
    {"beginRendering": {"surfaceId": "s1"}},
    {"surfaceUpdate": {"surfaceId": "s1", "components": []}},
    {
        "beginRendering": {"surfaceId": "s1", "root": "r"},
        "deleteSurface": {"surfaceId": "s1"},
    },
    {"unknownType": {"surfaceId": "s1"}},
]

# ---------------------------------------------------------------------------
# Valid client-to-server payloads
# ---------------------------------------------------------------------------

VALID_CLIENT_EVENTS: list[dict[str, Any]] = [
    {
        "userAction": {
            "name": "click",
            "surfaceId": "s1",
            "sourceComponentId": "btn-1",
            "timestamp": "2026-03-12T10:00:00Z",
            "context": {},
        },
    },
    {
        "userAction": {
            "name": "submit",
            "surfaceId": "s1",
            "sourceComponentId": "btn-2",
            "timestamp": "2026-03-12T10:00:00Z",
            "context": {"field": "value"},
        },
    },
    {
        "error": {"message": "render failed", "code": 500},
    },
]

# ---------------------------------------------------------------------------
# Invalid client-to-server payloads
# ---------------------------------------------------------------------------

INVALID_CLIENT_EVENTS: list[dict[str, Any]] = [
    {},
    {"userAction": {"name": "click"}},
    {
        "userAction": {
            "name": "click",
            "surfaceId": "s1",
            "sourceComponentId": "btn-1",
            "timestamp": "2026-03-12T10:00:00Z",
            "context": {},
        },
        "error": {"message": "oops"},
    },
]

# ---------------------------------------------------------------------------
# Catalog component payloads (validated structurally)
# ---------------------------------------------------------------------------

VALID_COMPONENTS: dict[str, dict[str, Any]] = {
    "Text": {"text": {"literalString": "hello"}, "usageHint": "h1"},
    "Image": {"url": {"path": "/img/url"}, "fit": "cover", "usageHint": "avatar"},
    "Icon": {"name": {"literalString": "home"}},
    "Video": {"url": {"literalString": "https://example.com/video.mp4"}},
    "AudioPlayer": {"url": {"literalString": "https://example.com/audio.mp3"}},
    "Row": {"children": {"explicitList": ["a", "b"]}, "distribution": "center"},
    "Column": {"children": {"template": {"componentId": "c1", "dataBinding": "/list"}}},
    "List": {"children": {"explicitList": ["x"]}, "direction": "horizontal"},
    "Card": {"child": "inner"},
    "Tabs": {"tabItems": [{"title": {"literalString": "Tab 1"}, "child": "content"}]},
    "Divider": {"axis": "horizontal"},
    "Modal": {"entryPointChild": "trigger", "contentChild": "body"},
    "Button": {"child": "label", "action": {"name": "go"}},
    "CheckBox": {"label": {"literalString": "Accept"}, "value": {"literalBoolean": False}},
    "TextField": {"label": {"literalString": "Name"}},
    "DateTimeInput": {"value": {"path": "/date"}},
    "MultipleChoice": {
        "selections": {"literalArray": ["a"]},
        "options": [{"label": {"literalString": "A"}, "value": "a"}],
    },
    "Slider": {"value": {"literalNumber": 50}, "minValue": 0, "maxValue": 100},
}


class TestServerToClientConformance:
    """Pydantic models and JSON schema must agree on server-to-client messages."""

    @pytest.mark.parametrize("payload", VALID_SERVER_MESSAGES)
    def test_valid_accepted_by_both(self, payload: dict[str, Any]) -> None:
        assert _json_schema_valid(SERVER_SCHEMA, payload), (
            f"JSON schema rejected valid payload: {payload}"
        )
        assert _pydantic_valid_message(payload), (
            f"Pydantic rejected valid payload: {payload}"
        )

    @pytest.mark.parametrize("payload", INVALID_SERVER_MESSAGES)
    def test_invalid_rejected_by_pydantic(self, payload: dict[str, Any]) -> None:
        assert not _pydantic_valid_message(payload), (
            f"Pydantic accepted invalid payload: {payload}"
        )


class TestClientToServerConformance:
    """Pydantic models and JSON schema must agree on client-to-server events."""

    @pytest.mark.parametrize("payload", VALID_CLIENT_EVENTS)
    def test_valid_accepted_by_both(self, payload: dict[str, Any]) -> None:
        assert _json_schema_valid(CLIENT_SCHEMA, payload), (
            f"JSON schema rejected valid payload: {payload}"
        )
        assert _pydantic_valid_event(payload), (
            f"Pydantic rejected valid payload: {payload}"
        )

    @pytest.mark.parametrize("payload", INVALID_CLIENT_EVENTS)
    def test_invalid_rejected_by_pydantic(self, payload: dict[str, Any]) -> None:
        assert not _pydantic_valid_event(payload), (
            f"Pydantic accepted invalid payload: {payload}"
        )


class TestCatalogConformance:
    """Catalog component schemas and Pydantic models must define the same components."""

    def test_catalog_component_names_match(self) -> None:
        from crewai.a2a.extensions.a2ui.catalog import STANDARD_CATALOG_COMPONENTS

        schema_components = set(CATALOG_SCHEMA["components"].keys())
        assert schema_components == STANDARD_CATALOG_COMPONENTS

    @pytest.mark.parametrize(
        "name,props",
        list(VALID_COMPONENTS.items()),
    )
    def test_valid_component_accepted_by_catalog_schema(
        self, name: str, props: dict[str, Any]
    ) -> None:
        component_schema = CATALOG_SCHEMA["components"][name]
        assert _json_schema_valid(component_schema, props), (
            f"Catalog schema rejected valid {name}: {props}"
        )

    @pytest.mark.parametrize(
        "name,props",
        list(VALID_COMPONENTS.items()),
    )
    def test_valid_component_accepted_by_pydantic(
        self, name: str, props: dict[str, Any]
    ) -> None:
        model_cls = getattr(catalog, name)
        try:
            model_cls.model_validate(props)
        except Exception as exc:
            pytest.fail(f"Pydantic {name} rejected valid props: {exc}")

    def test_catalog_required_fields_match(self) -> None:
        """Required fields in the JSON schema match non-optional Pydantic fields."""
        for comp_name, comp_schema in CATALOG_SCHEMA["components"].items():
            schema_required = set(comp_schema.get("required", []))
            model_cls = getattr(catalog, comp_name)
            pydantic_required = {
                info.alias or field_name
                for field_name, info in model_cls.model_fields.items()
                if info.is_required()
            }
            assert schema_required == pydantic_required, (
                f"{comp_name}: schema requires {schema_required}, "
                f"Pydantic requires {pydantic_required}"
            )

    def test_catalog_fields_match(self) -> None:
        """Field names in JSON schema match Pydantic model aliases."""
        for comp_name, comp_schema in CATALOG_SCHEMA["components"].items():
            schema_fields = set(comp_schema.get("properties", {}).keys())
            model_cls = getattr(catalog, comp_name)
            pydantic_fields = {
                info.alias or field_name
                for field_name, info in model_cls.model_fields.items()
            }
            assert schema_fields == pydantic_fields, (
                f"{comp_name}: schema has {schema_fields}, "
                f"Pydantic has {pydantic_fields}"
            )
