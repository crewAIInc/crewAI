import os
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar


try:
    from linkup import LinkupClient

    LINKUP_AVAILABLE = True
except ImportError:
    LINKUP_AVAILABLE = False
    LinkupClient = Any  # type: ignore[misc,assignment]  # type placeholder when package is not available

from pydantic import Field, PrivateAttr


class LinkupSearchTool(BaseTool):
    name: str = "Linkup Search Tool"
    description: str = (
        "Performs an API call to Linkup to retrieve contextual information."
    )
    _client: LinkupClient = PrivateAttr()  # type: ignore
    package_dependencies: list[str] = Field(default_factory=lambda: ["linkup-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="LINKUP_API_KEY", description="API key for Linkup", required=True
            ),
        ]
    )

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the tool with an API key."""
        super().__init__()  # type: ignore[call-arg]
        try:
            from linkup import LinkupClient
        except ImportError:
            import click

            if click.confirm(
                "You are missing the 'linkup-sdk' package. Would you like to install it?"
            ):
                import subprocess

                subprocess.run(["uv", "add", "linkup-sdk"], check=True)  # noqa: S607
                from linkup import LinkupClient

            else:
                raise ImportError(
                    "The 'linkup-sdk' package is required to use the LinkupSearchTool. "
                    "Please install it with: uv add linkup-sdk"
                ) from None
        self._client = LinkupClient(api_key=api_key or os.getenv("LINKUP_API_KEY"))

    def _run(
        self,
        query: str,
        depth: Literal["standard", "deep"] = "standard",
        output_type: Literal[
            "searchResults", "sourcedAnswer", "structured"
        ] = "searchResults",
    ) -> dict:
        """Executes a search using the Linkup API.

        :param query: The query to search for.
        :param depth: Search depth (default is "standard").
        :param output_type: Desired result type (default is "searchResults").
        :return: A dictionary containing the results or an error message.
        """
        try:
            response = self._client.search(
                query=query, depth=depth, output_type=output_type
            )
            results = [
                {"name": result.name, "url": result.url, "content": result.content}
                for result in response.results
            ]
            return {"success": True, "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}
