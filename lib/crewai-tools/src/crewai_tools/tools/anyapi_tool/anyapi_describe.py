from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.anyapi_tool.anyapi_base import AnyApiToolBase


class AnyApiDescribeToolSchema(BaseModel):
    """Input for AnyApiDescribeTool."""

    slug: str = Field(
        ...,
        description=(
            "The slug of the AnyAPI endpoint to describe, as returned by 'AnyAPI "
            "catalog search', for example 'instagram.profile'."
        ),
    )


class AnyApiDescribeTool(AnyApiToolBase):
    """Fetch one AnyAPI endpoint's input schema and price before running it."""

    name: str = "AnyAPI endpoint schema"
    description: str = (
        "Fetch the full definition of one AnyAPI endpoint by slug: its input JSON "
        "Schema, its output JSON Schema and its USD price per request. Call this after "
        "'AnyAPI catalog search' and before the first 'AnyAPI endpoint run' on a slug. "
        "Build the run input only from the schema this returns, because every AnyAPI "
        "input schema is strict and an invented field name fails the call."
    )
    args_schema: type[BaseModel] = AnyApiDescribeToolSchema

    def _run(self, slug: str, **_: Any) -> str:
        try:
            entry = self._client.describe(slug)
        except (self._anyapi.AnyAPIError, ValueError) as exc:
            return f"AnyAPI could not describe '{slug}': {exc}"

        return self._as_json(entry)
