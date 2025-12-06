import json
import os
from typing import Annotated, Any, Literal
import warnings

from crewai.tools import BaseTool, EnvVar
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


load_dotenv()
try:
    from parallel import Parallel

    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


class ParallelSearchInput(BaseModel):
    """Input schema for ParallelSearchTool using the Search API.

    At least one of objective or search_queries is required.
    """

    objective: str | None = Field(
        None,
        description="Natural-language goal for the web research (<=5000 chars)",
        max_length=5000,
    )
    search_queries: list[Annotated[str, Field(max_length=200)]] | None = Field(
        default=None,
        description="Optional list of keyword queries (<=5 items, each <=200 chars)",
        min_length=1,
        max_length=5,
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of search results to return (1-20). If not provided, uses tool default.",
    )
    source_policy: dict[str, Any] | None = Field(
        default=None,
        description="Source policy for domain inclusion/exclusion. Example: {'include': ['example.com']} or {'exclude': ['spam.com']}",
    )


class ParallelSearchTool(BaseTool):
    """Tool that uses the Parallel Search API to perform web searches.

    Attributes:
        client: An instance of Parallel client.
        name: The name of the tool.
        description: A description of the tool's purpose.
        args_schema: The schema for the tool's arguments.
        api_key: The Parallel API key.
        mode: Search mode ('one-shot' or 'agentic').
        max_results: Maximum number of results to return.
        excerpts: Excerpt configuration for result length.
        fetch_policy: Content freshness control.
        source_policy: Domain inclusion/exclusion policy.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    client: Any = None
    name: str = "Parallel Web Search"
    description: str = (
        "A tool that performs web searches using the Parallel Search API. "
        "Returns ranked results with compressed excerpts optimized for LLMs."
    )
    args_schema: type[BaseModel] = ParallelSearchInput
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("PARALLEL_API_KEY"),
        description="The Parallel API key. If not provided, it will be loaded from the environment variable PARALLEL_API_KEY.",
    )
    mode: Literal["one-shot", "agentic"] | None = Field(
        default=None,
        description=(
            "Search mode: 'one-shot' (comprehensive results, default) or "
            "'agentic' (concise, token-efficient for multi-step workflows)"
        ),
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Maximum number of search results to return (1-20)",
    )
    excerpts: dict[str, int] | None = Field(
        default=None,
        description="Excerpt configuration: {'max_chars_per_result': 10000, 'max_chars_total': 50000}",
    )
    fetch_policy: dict[str, int] | None = Field(
        default=None,
        description="Content freshness control: {'max_age_seconds': 3600}",
    )
    source_policy: dict[str, Any] | None = Field(
        default=None,
        description="Source policy for domain inclusion/exclusion",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["parallel-web"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="PARALLEL_API_KEY",
                description="API key for Parallel search service",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not PARALLEL_AVAILABLE:
            raise ImportError(
                "Missing optional dependency 'parallel-web'. Install with:\n"
                "  uv add crewai-tools --extra parallel-web\n"
                "or\n"
                "  pip install parallel-web"
            )

        if "PARALLEL_API_KEY" not in os.environ and not kwargs.get("api_key"):
            raise ValueError(
                "Environment variable PARALLEL_API_KEY is required for ParallelSearchTool. "
                "Set it with: export PARALLEL_API_KEY='your_api_key'"
            )

        self.client = Parallel(api_key=self.api_key)

    def _run(
        self,
        objective: str | None = None,
        search_queries: list[str] | None = None,
        max_results: int | None = None,
        source_policy: dict[str, Any] | None = None,
        # Deprecated parameters for backwards compatibility
        processor: str | None = None,
        max_chars_per_result: int | None = None,
        **_: Any,
    ) -> str:
        """Synchronously performs a search using the Parallel API.

        Args:
            objective: Natural-language goal for the web research.
            search_queries: Optional list of keyword queries.
            max_results: Maximum results to return. Overrides init value if provided.
            source_policy: Domain inclusion/exclusion policy. Overrides init value if provided.
            processor: DEPRECATED - no longer used.
            max_chars_per_result: DEPRECATED - use excerpts config instead.

        Returns:
            A JSON string containing the search results.
        """
        if not self.client:
            raise ValueError(
                "Parallel client is not initialized. Ensure 'parallel-web' is installed and API key is set."
            )

        if not objective and not search_queries:
            return "Error: Provide at least one of 'objective' or 'search_queries'"

        # Handle deprecated parameters
        excerpts = self._handle_deprecated_params(processor, max_chars_per_result)

        # Use runtime values if provided, otherwise fall back to init values
        effective_max_results = (
            max_results if max_results is not None else self.max_results
        )
        effective_source_policy = (
            source_policy if source_policy is not None else self.source_policy
        )

        search_params = self._build_search_params(
            objective,
            search_queries,
            excerpts,
            effective_max_results,
            effective_source_policy,
        )

        try:
            response = self.client.beta.search(**search_params)
            return self._format_output(response)
        except Exception as exc:
            return f"Parallel Search API error: {exc}"

    def _handle_deprecated_params(
        self,
        processor: str | None,
        max_chars_per_result: int | None,
    ) -> dict[str, int] | None:
        """Handle deprecated parameters with warnings."""
        excerpts = self.excerpts

        if processor is not None:
            warnings.warn(
                "The 'processor' parameter is deprecated and will be ignored. "
                "Use 'mode' instead when initializing the tool.",
                DeprecationWarning,
                stacklevel=3,
            )

        if max_chars_per_result is not None:
            warnings.warn(
                "The 'max_chars_per_result' parameter is deprecated. "
                "Use 'excerpts={\"max_chars_per_result\": N}' when initializing the tool.",
                DeprecationWarning,
                stacklevel=3,
            )
            # Map to new excerpts parameter for backwards compatibility
            if excerpts is None:
                excerpts = {"max_chars_per_result": max_chars_per_result}
            elif "max_chars_per_result" not in excerpts:
                excerpts = {**excerpts, "max_chars_per_result": max_chars_per_result}

        return excerpts

    def _build_search_params(
        self,
        objective: str | None,
        search_queries: list[str] | None,
        excerpts: dict[str, int] | None,
        max_results: int,
        source_policy: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Build search parameters dictionary."""
        search_params: dict[str, Any] = {"max_results": max_results}

        if objective is not None:
            search_params["objective"] = objective
        if search_queries is not None:
            search_params["search_queries"] = search_queries
        if self.mode is not None:
            search_params["mode"] = self.mode
        if excerpts is not None:
            search_params["excerpts"] = excerpts
        if self.fetch_policy is not None:
            search_params["fetch_policy"] = self.fetch_policy
        if source_policy is not None:
            search_params["source_policy"] = source_policy

        return search_params

    def _format_output(self, result: Any) -> str:
        """Format the search response as a JSON string."""
        try:
            # Handle SDK response object - convert to dict if needed
            if hasattr(result, "model_dump"):
                data = result.model_dump()
            elif hasattr(result, "to_dict"):
                data = result.to_dict()
            elif hasattr(result, "__dict__"):
                data = dict(result.__dict__)
            else:
                data = result

            return json.dumps(data or {}, ensure_ascii=False, default=str)
        except Exception:
            return str(result or {})
