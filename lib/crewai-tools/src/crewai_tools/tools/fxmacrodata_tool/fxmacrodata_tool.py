import json
import logging
from os import getenv
from typing import Any, ClassVar, Literal
import urllib.error
import urllib.parse
import urllib.request

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field


logger = logging.getLogger(__file__)

Dataset = Literal[
    "catalogue",
    "latest",
    "history",
    "calendar",
    "press_releases",
    "fx_rate",
    "rate_differential",
    "cot",
    "commodities",
    "market_sessions",
    "risk_sentiment",
]


class FXMacroDataToolInput(BaseModel):
    dataset: Dataset = Field(
        ...,
        description=(
            "Which dataset to read. 'catalogue' lists the indicator slugs available for a "
            "currency and is the right first call when the slug is unknown. 'latest' returns "
            "the most recent print of every indicator for one currency in a single request. "
            "'history' returns one indicator's published history. 'calendar' returns upcoming "
            "scheduled releases. 'press_releases' returns official central-bank headlines. "
            "'fx_rate' and 'rate_differential' take base and quote. 'cot' takes a currency. "
            "'commodities', 'market_sessions' and 'risk_sentiment' take no arguments."
        ),
    )
    currency: str = Field(
        "USD",
        description="Three-letter currency code, for example 'USD', 'EUR' or 'JPY'.",
    )
    indicator: str | None = Field(
        None,
        description=(
            "Indicator slug, required for dataset='history'. Get valid slugs from "
            "dataset='catalogue', for example 'inflation', 'non_farm_payrolls', 'policy_rate'."
        ),
    )
    base: str | None = Field(
        None, description="Base currency for dataset='fx_rate' or 'rate_differential'."
    )
    quote: str | None = Field(
        None, description="Quote currency for dataset='fx_rate' or 'rate_differential'."
    )
    start_date: str | None = Field(
        None,
        description="Optional ISO start date for dataset='history', e.g. '2024-01-01'.",
    )
    end_date: str | None = Field(
        None, description="Optional ISO end date for dataset='history'."
    )
    limit: int = Field(
        20,
        ge=1,
        le=100,
        description="Maximum rows to return; the API caps this at 100.",
    )


class FXMacroDataTool(BaseTool):
    """Official-source macroeconomic, FX and central-bank data for 18 currencies.

    FXMacroData aggregates official publishers - statistical agencies, central
    banks and exchanges - behind one contract, so answering "what did US core
    inflation print at, and when is the next release" does not require knowing
    which of eighteen publishers to call or how each formats its data. Every
    observation carries the instant it was published, so an agent can reason
    about what was knowable at a point in time rather than only about now.

    USD works without an API key. A key widens the history window and unlocks
    the other seventeen currencies plus FX rates, rate differentials, COT
    positioning and commodities.
    """

    BASE_API_URL: ClassVar[str] = "https://api.fxmacrodata.com/v1"
    REQUEST_TIMEOUT: ClassVar[int] = 30

    name: str = "FXMacroData Macroeconomic and FX Data"
    description: str = (
        "Reads official macroeconomic releases, release calendars, FX reference rates, "
        "rate differentials, COT positioning and commodity prices across 18 currencies. "
        "Use dataset='catalogue' first if you do not know an indicator slug."
    )
    args_schema: type[BaseModel] = FXMacroDataToolInput
    model_config = ConfigDict(extra="allow")
    package_dependencies: list[str] = Field(default_factory=lambda: ["pydantic"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="FXMACRODATA_API_KEY",
                description=(
                    "FXMacroData API key. Optional: USD data is public. Required for the "
                    "other currencies and for FX, COT and commodity data."
                ),
                required=False,
                default=None,
            )
        ]
    )
    api_key: str | None = None
    base_url: str = BASE_API_URL

    def _run(
        self,
        dataset: str,
        currency: str = "USD",
        indicator: str | None = None,
        base: str | None = None,
        quote: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 20,
    ) -> str:
        try:
            args = FXMacroDataToolInput(
                dataset=dataset,
                currency=currency,
                indicator=indicator,
                base=base,
                quote=quote,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
        except Exception as exc:
            return f"Invalid arguments for FXMacroData: {exc}"

        try:
            endpoint, params = self._resolve(args)
        except ValueError as exc:
            return str(exc)

        return self._request(endpoint, params)

    def _resolve(self, args: FXMacroDataToolInput) -> tuple[str, dict[str, Any]]:
        """Map the requested dataset onto an endpoint path and query parameters."""
        currency = args.currency.lower()
        paged = {"limit": args.limit}

        if args.dataset == "catalogue":
            return f"data_catalogue/{currency}", {}
        if args.dataset == "latest":
            return f"announcements/{currency}/latest", {}
        if args.dataset == "history":
            if not args.indicator:
                raise ValueError(
                    "dataset='history' needs an indicator slug. Call dataset='catalogue' "
                    "first to list the slugs available for this currency."
                )
            params: dict[str, Any] = dict(paged)
            if args.start_date:
                params["start_date"] = args.start_date
            if args.end_date:
                params["end_date"] = args.end_date
            return f"announcements/{currency}/{args.indicator}", params
        if args.dataset == "calendar":
            return f"calendar/{currency}", paged
        if args.dataset == "press_releases":
            return f"press-releases/{currency}", paged
        if args.dataset in ("fx_rate", "rate_differential"):
            if not args.base or not args.quote:
                raise ValueError(
                    f"dataset='{args.dataset}' needs both base and quote currencies."
                )
            path = "forex" if args.dataset == "fx_rate" else "rate_differentials"
            return f"{path}/{args.base.lower()}/{args.quote.lower()}", paged
        if args.dataset == "cot":
            return f"cot/{currency}", paged
        if args.dataset == "commodities":
            return "commodities/latest", {}
        if args.dataset == "market_sessions":
            return "market_sessions", {}
        return "risk_sentiment", {}

    def _request(self, endpoint: str, params: dict[str, Any]) -> str:
        url = f"{self.base_url.rstrip('/')}/{endpoint}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"

        headers = {"Accept": "application/json"}
        api_key = self.api_key or getenv("FXMACRODATA_API_KEY")
        if api_key:
            # Sent as a header rather than a query parameter so the key does not
            # end up in proxy or server access logs.
            headers["X-API-Key"] = api_key

        # base_url is settable, so pin the scheme before opening: urlopen would
        # otherwise honour file:// and read from the local filesystem.
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return f"Refusing to call FXMacroData over an unsupported scheme: {parsed.scheme or 'none'}"

        request = urllib.request.Request(url, headers=headers)  # noqa: S310
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=self.REQUEST_TIMEOUT
            ) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                logger.info(
                    "FXMacroData denied %s (HTTP %s): API key required",
                    endpoint,
                    exc.code,
                )
                return (
                    f"FXMacroData denied the request to {endpoint} (HTTP {exc.code}). This data "
                    "requires an API key; USD macro data is available without one."
                )
            logger.error(
                "FXMacroData request to %s failed: HTTP %s", endpoint, exc.code
            )
            return f"FXMacroData request to {endpoint} failed with HTTP {exc.code}."
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            logger.error("FXMacroData request to %s failed: %s", endpoint, exc)
            return f"FXMacroData request to {endpoint} failed: {exc}"

        try:
            json.loads(payload)
        except json.JSONDecodeError:
            return f"FXMacroData response from {endpoint} was not valid JSON."
        return payload
