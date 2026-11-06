import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from crewai.tools.tool_failure import ToolFailure
from pydantic import Field, PrivateAttr


__all__ = ["DecodoBaseTool"]

API_KEY_ENV_VAR = "DECODO_API_KEY"
INTEGRATION_HEADER = "crewai-python"


class DecodoBaseTool(BaseTool):
    """Base class for the Decodo Web Scraping API tools.

    Holds what every Decodo tool shares: the API client, and the translation of
    a Web Scraping API response into either the scraped content or a
    :class:`ToolFailure`.

    Get an API key from your Web Scraping API subscription on the Decodo dashboard:
    https://dashboard.decodo.com/web-data/playground
    """

    package_dependencies: list[str] = Field(default_factory=lambda: ["decodo-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name=API_KEY_ENV_VAR,
                description="API key for the Decodo Web Scraping API",
                required=True,
            )
        ]
    )

    _client: Any = PrivateAttr()

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        api_key = (api_key or os.environ.get(API_KEY_ENV_VAR) or "").strip()
        if not api_key:
            raise ValueError(
                f"Pass api_key or set the {API_KEY_ENV_VAR} environment variable."
            )
        self._client = self._build_client(api_key)

    @staticmethod
    def _build_client(api_key: str) -> Any:
        try:
            import decodo
        except ImportError:
            import click

            if not click.confirm(
                "You are missing the 'decodo-sdk' package. Would you like to install it?"
            ):
                raise ImportError(
                    "`decodo-sdk` package not found, please run `uv add decodo-sdk`"
                ) from None

            import importlib
            import subprocess

            try:
                subprocess.run(["uv", "add", "decodo-sdk"], check=True)  # noqa: S607
            except (subprocess.CalledProcessError, OSError) as e:
                raise ImportError("Failed to install decodo-sdk package") from e

            decodo = importlib.import_module("decodo")

        return decodo.DecodoClient(
            decodo.DecodoConfig(
                web_scraping_api=decodo.WebScrapingApiConfig(
                    api_key=api_key, integration_header=INTEGRATION_HEADER
                ),
            )
        )

    def _scrape(self, params: Any) -> Any:
        """Return the first result's content, or a ToolFailure the agent can act on."""
        import decodo
        import httpx

        try:
            response = self._client.web_scraping_api.scrape(params)
        except decodo.AuthenticationError as e:
            return ToolFailure(
                message=f"Decodo rejected the API key ({e}). Check {API_KEY_ENV_VAR}.",
                code="authentication_failed",
            )
        except decodo.RateLimitError as e:
            return ToolFailure(
                message=f"Decodo rate limit reached: {e}",
                code="rate_limited",
                retryable=True,
            )
        except decodo.ValidationError as e:
            return ToolFailure(
                message=f"Decodo rejected the request parameters: {e}",
                code="invalid_params",
            )
        except decodo.TimeoutError as e:
            return ToolFailure(
                message=f"Decodo request timed out: {e}", code="timeout", retryable=True
            )
        except decodo.DecodoError as e:
            return ToolFailure(
                message=f"Decodo request failed: {e}",
                code=str(e.status_code),
                retryable=e.status_code >= 500,
            )
        # httpx quotes the rejected header, API key included, so the error itself
        # must not reach the agent.
        except (httpx.LocalProtocolError, UnicodeEncodeError):
            return ToolFailure(
                message=(
                    "The request to Decodo could not be sent. "
                    f"Check {API_KEY_ENV_VAR} for invalid characters."
                ),
                code="invalid_request",
            )
        except httpx.TransportError as e:
            return ToolFailure(
                message=f"Could not reach Decodo: {e}",
                code="network_error",
                retryable=True,
            )

        results = response.get("results") or []
        if not results:
            failed: dict[str, Any] = dict(response)
            if failed.get("message"):
                return ToolFailure(
                    message=f"Decodo could not scrape the target: {failed['message']}",
                    code=str(failed.get("status_code", "scrape_failed")),
                    retryable=True,
                )
            return ToolFailure(
                message="Decodo returned no results.", code="empty_response"
            )

        result = results[0]
        status_code = result.get("status_code")
        if isinstance(status_code, int) and not 200 <= status_code < 300:
            return ToolFailure(
                message=(
                    "Decodo could not retrieve the page: the target responded "
                    f"with status {status_code}."
                ),
                code=str(status_code),
                retryable=status_code == 429 or status_code >= 500,
            )

        content = result.get("content")
        if content is None or content == "":
            return ToolFailure(
                message="Decodo returned a result with no content.",
                code="empty_content",
            )
        return content
