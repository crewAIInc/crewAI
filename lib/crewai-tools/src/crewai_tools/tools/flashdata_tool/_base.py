"""Shared authentication and source contracts for FlashData tools."""

import asyncio
import os
from typing import Any, ClassVar
from urllib.parse import urlsplit

from crewai.tools import BaseTool, EnvVar
from crewai.types.callback import SerializableCallable
from pydantic import BaseModel, Field, StrictBool, ValidationError, field_validator

from crewai_tools.tools.flashdata_tool._client import Client, FlashDataError
from crewai_tools.tools.flashdata_tool._inputs import YouTubeDownloadInput
from crewai_tools.tools.flashdata_tool._storage import storage_config


def no_cache(_args: Any = None, _result: Any = None) -> bool:
    """Keep live results and asynchronous job state out of the tool cache."""
    return False


class FlashDataBaseTool(BaseTool):
    input_model: ClassVar[type[BaseModel]]
    api_key_env: str = Field(default="FLASHDATA_API_KEY", min_length=1)
    cache_function: SerializableCallable = Field(default=no_cache, exclude=True)
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="FLASHDATA_API_KEY",
                description="FlashData API key. Requests use paid FlashData Credits.",
                required=True,
            )
        ]
    )

    @field_validator("args_schema", mode="before")
    @classmethod
    def preserve_source_schema(cls, _value: Any) -> type[BaseModel]:
        # Generic JSON schema restoration drops custom input validators and aliases.
        return cls.input_model

    def _validate_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            return self.input_model.model_validate(kwargs).model_dump(
                exclude_none=True, by_alias=True
            )
        except ValidationError:
            raise FlashDataError(
                "Invalid source inputs. Check the tool input schema."
            ) from None

    def _client(self) -> Client:
        value = os.environ.get(self.api_key_env, "").strip()
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise FlashDataError(
                "Set a valid API key in the configured environment variable."
            )
        return Client(value)

    async def _arun(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self._run, **kwargs)


class FlashDataRealtimeTool(FlashDataBaseTool):
    source: ClassVar[str]

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        data = self._validate_kwargs(kwargs)
        primary = next(key for key in ("query", "video_id", "handle") if key in data)
        return self._client().query(
            {"source": self.source, primary: data.pop(primary), "params": data}
        )


class FlashDataDownloadSubmitTool(FlashDataBaseTool):
    input_model: ClassVar[type[BaseModel]] = YouTubeDownloadInput
    storage_env: str = Field(default="FLASHDATA_DOWNLOAD_STORAGE", min_length=1)
    allow_submit: StrictBool = False

    def _run(self, **kwargs: Any) -> dict[str, Any]:
        if not self.allow_submit:
            raise FlashDataError(
                "Configure allow_submit=True only after authorizing a paid customer-storage "
                "write. Use the status or result tool to inspect an existing job."
            )
        data = self._validate_kwargs(kwargs)
        storage = storage_config(os.environ.get(self.storage_env, ""))
        if storage.get("endpoint_url"):
            try:
                url = urlsplit(storage["endpoint_url"])
                if (
                    url.scheme != "https"
                    or not url.hostname
                    or url.username
                    or url.password
                    or url.query
                    or url.fragment
                ):
                    raise ValueError()
                _ = url.port
            except ValueError:
                raise FlashDataError(
                    "Storage endpoint must be an HTTPS URL without credentials."
                ) from None
        return self._client().submit(
            {
                "source": "youtube_download",
                "video_id": data.pop("video_id"),
                "params": {**data, "storage": storage},
            }
        )
