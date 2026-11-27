from crewai_tools.tools.hyperbrowser_load_tool.hyperbrowser_load_tool import (
    HyperbrowserLoadToolSchema,
)


def test_params_is_optional() -> None:
    """`params` is typed as optional, so the tool call must not require it."""
    schema = HyperbrowserLoadToolSchema(url="https://example.com", operation="scrape")

    assert schema.params is None
    assert "params" not in HyperbrowserLoadToolSchema.model_json_schema()["required"]
