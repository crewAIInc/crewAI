from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.anyapi_tool.anyapi_base import AnyApiToolBase


# AnyAPI answers 402 with three codes and only one of them is about the wallet:
# a spend cap on the key or on an authorized connection stops a call whatever the
# balance is, so adding funds would not help.
CAP_CODES = frozenset({"key_cap_exceeded", "grant_cap_exceeded"})


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
            return f"AnyAPI run failed for '{slug}': {exc}{self._spend_hint(exc)}"
        except (self._anyapi.AnyAPIError, ValueError) as exc:
            return f"AnyAPI run failed for '{slug}': {exc}"

        return self._as_json(result)

    def _spend_hint(self, exc: Any) -> str:
        """Point at the limit that stopped the call: a spend cap, or the wallet."""
        if getattr(exc, "code", None) in CAP_CODES:
            return (
                " A spend limit on this key, not the wallet balance, stopped the "
                "call. Raise it at https://getanyapi.com/dashboard."
            )

        try:
            balance = self._client.balance()
        except (self._anyapi.AnyAPIError, ValueError):
            return ""

        return (
            f" The wallet holds ${balance.usd} USD. "
            "Add funds at https://getanyapi.com/dashboard."
        )
