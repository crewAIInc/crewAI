from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.anyapi_tool.anyapi_base import AnyApiToolBase


class AnyApiRunToolSchema(BaseModel):
    """Input for AnyApiRunTool."""

    slug: str = Field(
        ...,
        description=(
            "The slug of the AnyAPI endpoint to run, for example 'instagram.profile'."
        ),
    )
    input: dict[str, Any] = Field(
        ...,
        description=(
            "The endpoint input, built from the input schema returned by 'AnyAPI "
            "endpoint schema'. The schema is strict, so an unknown field fails the "
            'call. Example for the instagram.profile endpoint: {"handle": "nasa"}.'
        ),
    )


class AnyApiRunTool(AnyApiToolBase):
    """Execute one AnyAPI endpoint and return its normalized output."""

    name: str = "AnyAPI endpoint run"
    description: str = (
        "Run one AnyAPI endpoint by slug with a JSON input and return its normalized "
        "output. Use 'AnyAPI catalog search' first to pick a slug, then 'AnyAPI "
        "endpoint schema' to read that slug's input schema, and only then run it: the "
        "input schema is strict, so an invented field name fails the call. The response "
        "reports what the call cost in USD as costUsd. Failed calls are never charged, "
        "because AnyAPI fails over across sources automatically under one price."
    )
    args_schema: type[BaseModel] = AnyApiRunToolSchema

    def _run(self, slug: str, input: dict[str, Any], **_: Any) -> str:
        try:
            result = self._client.run(slug=slug, input=input)
        except self._anyapi.InsufficientBalanceError as exc:
            return f"AnyAPI run failed for '{slug}': {exc}{self._balance_hint()}"
        except self._anyapi.AnyAPIError as exc:
            return f"AnyAPI run failed for '{slug}': {exc}"

        return self._as_json(result)

    def _balance_hint(self) -> str:
        """Report the wallet balance so an agent knows how much is left."""
        try:
            balance = self._client.balance()
        except self._anyapi.AnyAPIError:
            return ""

        return (
            f" The wallet holds ${balance.usd} USD. "
            "Add funds at https://getanyapi.com/dashboard."
        )
