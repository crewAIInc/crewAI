from collections.abc import Callable, Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
import json
import logging
import os
from platform import architecture, python_version
import re
from typing import TYPE_CHECKING, Any

from crewai.tools import BaseTool, EnvVar
from crewai.tools.tool_failure import ToolFailure
from pydantic import ConfigDict, Field


__all__ = ["OxylabsBaseTool"]


_HTTP_ERROR_PATTERN = re.compile(
    r"(\d{3})\s+(?:Client|Server) Error:\s*(.+?)\s+for url", re.IGNORECASE
)


@contextmanager
def _captured_sdk_errors() -> Iterator[list[str]]:
    """Collect the error messages the oxylabs SDK only writes to its logger.

    The SDK catches transport and HTTP errors, logs them and hands back an
    empty response, so the cause is absent from the object we get back.
    Listening on its logger is the only way to tell the agent what actually
    went wrong. Nothing about the caller's logging setup is modified, so an
    application that has silenced the SDK still gets the generic failure.
    """
    messages: list[str] = []

    class _Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    sdk_logger = logging.getLogger("oxylabs")
    handler = _Collector(level=logging.ERROR)
    sdk_logger.addHandler(handler)
    try:
        yield messages
    finally:
        sdk_logger.removeHandler(handler)


def _api_detail(messages: list[str]) -> str | None:
    """Pull the API's own explanation out of a logged response body."""
    for message in messages:
        try:
            payload = json.loads(message)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            detail = payload.get("message")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
    return None


def _diagnose(messages: list[str]) -> tuple[str, str | None, bool] | None:
    """Summarize what the SDK logged as (description, code, retryable)."""
    reported = [m.strip() for m in messages if m and m.strip()]
    if not reported:
        return None

    joined = " | ".join(reported)

    http_error = _HTTP_ERROR_PATTERN.search(joined)
    if http_error:
        status = int(http_error.group(1))
        description = f"{status} {http_error.group(2).strip()}"
        detail = _api_detail(reported)
        if detail:
            description = f"{description} - {detail}"
        return description, str(status), status == 429 or status >= 500

    if "timed out" in joined.lower():
        return "the request timed out", "timeout", True

    # Anything else the SDK chose to log, e.g. a connection error.
    return joined[:300], None, False


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

    if TYPE_CHECKING:
        # Declared as its own model by every subclass, and enforced in
        # ``__init__``; annotated here only so this class can read it.
        config: Any

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
        # Resolved before the client is built, so a subclass that forgets the
        # field fails without first opening a session.
        config_field = type(self).model_fields.get("config")
        config_model = config_field.annotation if config_field else None
        if config_model is None:
            raise TypeError(
                f"{type(self).__name__} must declare a 'config' model field"
            )
        if config is None:
            config = config_model()

        if username is None or password is None:
            username, password = self._get_credentials_from_env()

        kwargs["oxylabs_api"] = self._build_client(username, password)

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
            except (subprocess.CalledProcessError, OSError) as e:
                # OSError covers uv itself being absent.
                raise ImportError("Failed to install oxylabs package") from e

            return importlib.import_module("oxylabs").RealtimeClient

        return RealtimeClient

    @classmethod
    def _build_client(cls, username: str, password: str) -> Any:
        realtime_client = cls._resolve_realtime_client()

        try:
            crewai_version = version("crewai")
        except PackageNotFoundError:
            # Only ever reported as telemetry; not worth failing construction.
            crewai_version = "unknown"

        bits, _ = architecture()
        return realtime_client(
            username=username,
            password=password,
            sdk_type=(
                f"oxylabs-crewai-sdk-python/"
                f"{crewai_version} "
                f"({python_version()}; {bits})"
            ),
        )

    def _scrape(self, scraper: Callable[..., Any], target: str) -> str | ToolFailure:
        """Run one scrape and turn the outcome into content or a failure."""
        with _captured_sdk_errors() as sdk_errors:
            response = scraper(target, **self.config.model_dump(exclude_none=True))

        return self._handle_response(response, sdk_errors)

    def _handle_response(
        self, response: Any, sdk_errors: list[str] | None = None
    ) -> str | ToolFailure:
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
            diagnosis = _diagnose(sdk_errors or [])
            if diagnosis:
                description, code, retryable = diagnosis
                return ToolFailure(
                    message=(
                        f"Oxylabs Web Scraper API rejected the request: {description}"
                    ),
                    code=code or "request_rejected",
                    retryable=retryable,
                )
            return ToolFailure(
                message=(
                    "Oxylabs Web Scraper API returned no results and reported no "
                    "error, so the request was rejected before any page was "
                    "scraped. Check that OXYLABS_USERNAME and OXYLABS_PASSWORD "
                    "are valid and that this tool's config options are accepted "
                    "for this source."
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

        content = getattr(result, "content", None)
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
