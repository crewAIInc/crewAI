import shutil
import subprocess
from typing import Any, List, Literal, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

_VALID_FORMATS = ("markdown", "text", "som", "links")

_INSTALL_MSG = (
    "plasmate is required for PlasmateWebsiteTool. "
    "Install it with: pip install plasmate\n"
    "Docs: https://plasmate.app"
)


def _find_plasmate() -> Optional[str]:
    path = shutil.which("plasmate")
    if path:
        return path
    try:
        import plasmate as _p  # noqa: F401
        return shutil.which("plasmate")
    except ImportError:
        return None


class FixedPlasmateWebsiteToolSchema(BaseModel):
    """Input schema when website_url is fixed at initialisation time."""


class PlasmateWebsiteToolSchema(FixedPlasmateWebsiteToolSchema):
    """Input schema for PlasmateWebsiteTool."""

    website_url: str = Field(..., description="The URL of the website to read.")


class PlasmateWebsiteTool(BaseTool):
    """Read a website and return compact, LLM-ready content via Plasmate.

    PlasmateWebsiteTool is a drop-in replacement for ``ScrapeWebsiteTool`` that
    uses `Plasmate <https://github.com/plasmate-labs/plasmate>`_ — an open-source
    Rust browser engine — instead of raw HTTP + BeautifulSoup.  Plasmate strips
    navigation, ads, cookie banners, and boilerplate before the agent ever sees
    the page, returning 10-100x fewer tokens than raw HTML.

    No API key is required.  Plasmate runs locally as a subprocess.

    Typical output sizes (measured across 45 real sites):

    * Average compression: **17.7x** over raw HTML
    * Peak compression: **77x** (TechCrunch)

    Example::

        from crewai_tools import PlasmateWebsiteTool

        # Dynamic — agent supplies the URL
        tool = PlasmateWebsiteTool()

        # Fixed — URL locked at init time (agent can't change it)
        tool = PlasmateWebsiteTool(
            website_url="https://docs.crewai.com",
            output_format="markdown",
        )

    Install Plasmate::

        pip install plasmate
    """

    name: str = "Read website content with Plasmate"
    description: str = (
        "Fetches a website URL and returns its content as compact, structured text "
        "using Plasmate — a lightweight local browser engine.  The output contains "
        "only the meaningful content of the page (headings, body text, links) with "
        "navigation menus, ads, and boilerplate removed, giving the agent a "
        "10-100x token reduction compared to raw HTML."
    )
    args_schema: Type[BaseModel] = PlasmateWebsiteToolSchema
    website_url: Optional[str] = None
    output_format: Literal["markdown", "text", "som", "links"] = "markdown"
    timeout: int = 30
    selector: Optional[str] = None
    extra_headers: Optional[dict] = None
    package_dependencies: List[str] = ["plasmate"]

    def __init__(
        self,
        website_url: Optional[str] = None,
        output_format: Literal["markdown", "text", "som", "links"] = "markdown",
        timeout: int = 30,
        selector: Optional[str] = None,
        extra_headers: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialise PlasmateWebsiteTool.

        Args:
            website_url: If provided, the URL is fixed and the agent cannot
                override it.  Useful for scoping a crew to a single data source.
            output_format: Content format returned to the agent.  One of
                ``"markdown"`` (default), ``"text"``, ``"som"`` (structured JSON),
                or ``"links"`` (extracted hyperlinks only).
            timeout: Per-request timeout in seconds.  Defaults to 30.
            selector: Optional ARIA role or CSS id to scope extraction to a
                specific page region (e.g. ``"main"`` or ``"#article-body"``).
            extra_headers: Optional HTTP headers forwarded with each request
                (e.g. ``{"Accept-Language": "en-US"}``).
        """
        super().__init__(**kwargs)
        if output_format not in _VALID_FORMATS:
            raise ValueError(
                f"output_format must be one of {_VALID_FORMATS}; got {output_format!r}"
            )
        self.output_format = output_format
        self.timeout = timeout
        self.selector = selector
        self.extra_headers = extra_headers or {}

        if website_url is not None:
            self.website_url = website_url
            self.description = (
                f"Fetches {website_url} and returns its content as compact, "
                f"structured {output_format} using Plasmate.  Navigation, ads, and "
                "boilerplate are stripped automatically."
            )
            self.args_schema = FixedPlasmateWebsiteToolSchema
            self._generate_description()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cmd(self, url: str) -> List[str]:
        plasmate_bin = _find_plasmate()
        if plasmate_bin is None:
            raise ImportError(_INSTALL_MSG)
        cmd = [
            plasmate_bin,
            "fetch",
            url,
            "--format", self.output_format,
            "--timeout", str(self.timeout * 1000),  # plasmate uses ms
        ]
        if self.selector:
            cmd += ["--selector", self.selector]
        for key, value in self.extra_headers.items():
            cmd += ["--header", f"{key}: {value}"]
        return cmd

    # ------------------------------------------------------------------
    # BaseTool interface
    # ------------------------------------------------------------------

    def _run(self, **kwargs: Any) -> str:
        url = kwargs.get("website_url", self.website_url)
        if not url:
            return "Error: no URL provided.  Pass a website_url argument."

        try:
            result = subprocess.run(
                self._build_cmd(url),
                capture_output=True,
                text=True,
                timeout=self.timeout + 5,
            )
        except subprocess.TimeoutExpired:
            return f"Error: request to {url} timed out after {self.timeout}s."
        except FileNotFoundError:
            return (
                "Error: plasmate binary not found.  "
                "Install it with: pip install plasmate"
            )

        if result.returncode != 0:
            return (
                f"Error fetching {url} (plasmate exited {result.returncode}): "
                f"{result.stderr[:300]}"
            )

        content = result.stdout.strip()
        if not content:
            return f"Warning: plasmate returned empty content for {url}."

        return content
