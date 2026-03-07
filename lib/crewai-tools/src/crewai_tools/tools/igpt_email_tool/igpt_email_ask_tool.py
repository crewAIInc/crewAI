import json
import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, PrivateAttr


def _stringify_igpt_response(response: Any) -> str:
    if isinstance(response, (dict, list)):
        return json.dumps(response, ensure_ascii=False)
    return str(response)


class IgptEmailAskInput(BaseModel):
    question: str = Field(
        ..., description="Natural language question about a user's email history."
    )
    output_format: Literal["text", "json", "schema"] = Field(
        default="text",
        description="Expected output format: text, json, or schema.",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Required when output_format='schema'. "
            "JSON schema that the answer must follow."
        ),
    )


class IgptEmailAskTool(BaseTool):
    name: str = "iGPT Email Ask"
    description: str = (
        "Ask natural-language questions over a user's email history and receive "
        "grounded answers with citations for decisions, commitments, action items, "
        "and unresolved threads."
    )
    args_schema: type[BaseModel] = IgptEmailAskInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["igptai"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="IGPT_API_KEY",
                description="API key for iGPT services",
                required=True,
            ),
            EnvVar(
                name="IGPT_API_USER",
                description="User identifier for iGPT email memory",
                required=True,
            ),
        ]
    )
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("IGPT_API_KEY"),
        description="API key for iGPT services",
    )
    user: str | None = Field(
        default_factory=lambda: os.getenv("IGPT_API_USER"),
        description="User identifier for iGPT email memory",
    )
    quality: str = Field(
        default="cef-1-normal",
        description="Quality preset used by iGPT recall endpoints.",
    )
    _client: Any = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.api_key:
            raise ValueError(
                "IGPT API key must be provided via constructor or IGPT_API_KEY environment variable."
            )
        if not self.user:
            raise ValueError(
                "IGPT user must be provided via constructor or IGPT_API_USER environment variable."
            )

        try:
            from igptai import IGPT  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'igptai' package is required to use IgptEmailAskTool. "
                "Install it with: uv add igptai"
            ) from exc

        self._client = IGPT(api_key=self.api_key, user=self.user)

    def _run(
        self,
        question: str,
        output_format: Literal["text", "json", "schema"] = "text",
        output_schema: dict[str, Any] | None = None,
    ) -> str:
        resolved_output_format: str | dict[str, Any] = output_format
        if output_format == "schema":
            if output_schema is None:
                raise ValueError(
                    "output_schema must be provided when output_format is 'schema'."
                )
            resolved_output_format = {"strict": True, "schema": output_schema}

        response = self._client.recall.ask(
            input=question,
            quality=self.quality,
            output_format=resolved_output_format,
        )
        return _stringify_igpt_response(response)
