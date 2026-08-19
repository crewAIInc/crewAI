from importlib.metadata import version
import json
import os
from platform import architecture, python_version
from typing import Any

from crewai.tools import BaseTool, EnvVar
from crewai.tools.tool_failure import ToolFailure
from pydantic import ConfigDict, Field


__all__ = ["OxylabsBaseTool"]


class OxylabsBaseTool(BaseTool):
    """Base class for the Oxylabs Web Scraper API tools.

    Holds what every Oxylabs tool shares: the credentialed ``RealtimeClient``,
    and the translation of a Web Scraper API response into either the scraped
    content or a :class:`ToolFailure`.

    Get Oxylabs account:
    https://dashboard.oxylabs.io/en
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    oxylabs_api: Any
    package_dependencies: list[str] = Field(default_factory=lambda: ["oxylabs"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OXYLABS_USERNAME",
                description="Username for Oxylabs",
                required=True,
            ),
            EnvVar(
                name="OXYLABS_PASSWORD",
                description="Password for Oxylabs",
                required=True,
            ),
        ]
    )

    def __init__(
        self,
        username: str | None = None,
        password: str | None = None,
        config: Any = None,
        **kwargs: Any,
    ) -> None:
        if username is None or password is None:
            username, password = self._get_credentials_from_env()

        kwargs["oxylabs_api"] = self._build_client(username, password)

        if config is None:
            # Every subclass declares its own ``config`` model; an empty one of
            # that model is the "no options set" default.
            config_model = type(self).model_fields["config"].annotation
            if config_model is None:
                raise TypeError(
                    f"{type(self).__name__} must declare a 'config' model field"
                )
            config = config_model()

        super().__init__(config=config, **kwargs)

    @staticmethod
    def _get_credentials_from_env() -> tuple[str, str]:
        username = os.environ.get("OXYLABS_USERNAME")
        password = os.environ.get("OXYLABS_PASSWORD")
        if not username or not password:
            raise ValueError(
                "You must pass oxylabs username and password when instantiating the tool "
                "or specify OXYLABS_USERNAME and OXYLABS_PASSWORD environment variables"
            )
        return username, password

    @staticmethod
    def _resolve_realtime_client() -> Any:
        try:
            from oxylabs import RealtimeClient  # type: ignore[import-untyped]
        except ImportError:
            import click

            if not click.confirm(
                "You are missing the 'oxylabs' package. Would you like to install it?"
            ):
                raise ImportError(
                    "`oxylabs` package not found, please run `uv add oxylabs`"
                ) from None

            import importlib
            import subprocess

            try:
                subprocess.run(["uv", "add", "oxylabs"], check=True)  # noqa: S607
            except subprocess.CalledProcessError as e:
                raise ImportError("Failed to install oxylabs package") from e

            return importlib.import_module("oxylabs").RealtimeClient

        return RealtimeClient

    @classmethod
    def _build_client(cls, username: str, password: str) -> Any:
        realtime_client = cls._resolve_realtime_client()

        bits, _ = architecture()
        return realtime_client(
            username=username,
            password=password,
            sdk_type=(
                f"oxylabs-crewai-sdk-python/"
                f"{version('crewai')} "
                f"({python_version()}; {bits})"
            ),
        )

    def _handle_response(self, response: Any) -> str | ToolFailure:
        """Return the scraped content, or report why there is none.

        The oxylabs SDK logs transport and validation errors and hands back an
        empty response rather than raising, so rejected requests -- wrong
        credentials, config the source does not accept, exhausted quota -- have
        to be recognised here instead of reaching the agent as an ``IndexError``
        on ``results[0]``. A non-2xx ``status_code`` on the result is the same
        situation one level down: the job ran, the page did not come back.
        """
        results = getattr(response, "results", None)
        if not results:
            return ToolFailure(
                message=(
                    "Oxylabs Web Scraper API returned no results, so the request was "
                    "rejected before any page was scraped. Check that "
                    "OXYLABS_USERNAME and OXYLABS_PASSWORD are valid and that this "
                    "tool's config options are accepted for this source. The oxylabs "
                    "package logs the underlying HTTP error on the "
                    "'oxylabs.internal.api' logger."
                ),
                code="empty_response",
            )

        result = results[0]

        try:
            status_code = int(result.status_code)
        except (AttributeError, TypeError, ValueError):
            status_code = None

        if status_code is not None and not 200 <= status_code < 300:
            return ToolFailure(
                message=(
                    f"Oxylabs Web Scraper API could not retrieve the page: the "
                    f"target responded with status {status_code}."
                ),
                code=str(status_code),
                retryable=status_code == 429 or status_code >= 500,
            )

        content = result.content
        if content is None:
            return ToolFailure(
                message=(
                    "Oxylabs Web Scraper API returned a result with no content. "
                    "The page may be empty, or the parser found nothing to extract."
                ),
                code="empty_content",
            )

        # ``parse``/``parsing_instructions`` results arrive as dicts or lists;
        # only unparsed HTML comes back as a string. ``str()`` on a list would
        # hand the agent a Python repr instead of JSON.
        if isinstance(content, str):
            return content

        try:
            return json.dumps(content)
        except (TypeError, ValueError):
            return str(content)
