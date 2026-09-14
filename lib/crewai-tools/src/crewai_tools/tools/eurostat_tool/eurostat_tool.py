import logging
from typing import Any, ClassVar, cast

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


logger = logging.getLogger(__file__)

# Friendly names for a handful of commonly requested indicators, mapped to
# their Eurostat dataset code and default dimension filters. Each entry was
# verified against the live Eurostat API before being added here.
KNOWN_INDICATORS: dict[str, dict[str, Any]] = {
    "unemployment_rate": {
        "dataset": "une_rt_m",
        "filters": {"sex": "T", "age": "TOTAL", "s_adj": "SA", "unit": "PC_ACT"},
        "description": "Monthly seasonally-adjusted unemployment rate (% of active population).",
    },
    "inflation_rate": {
        "dataset": "prc_hicp_manr",
        "filters": {"coicop": "CP00", "unit": "RCH_A"},
        "description": "Monthly HICP annual rate of change (inflation), all-items.",
    },
    "gdp": {
        "dataset": "nama_10_gdp",
        "filters": {"unit": "CLV10_MEUR", "na_item": "B1GQ"},
        "description": "Annual GDP at market prices, chain-linked volumes (million EUR, 2010 reference).",
    },
    "population": {
        "dataset": "demo_pjan",
        "filters": {"sex": "T", "age": "TOTAL"},
        "description": "Population on 1 January, total.",
    },
}


class EurostatToolInput(BaseModel):
    indicator: str | None = Field(
        None,
        description=(
            "A known indicator shortcut: one of 'unemployment_rate', 'inflation_rate', "
            "'gdp', 'population'. Provide either this or 'dataset_code', not both."
        ),
    )
    dataset_code: str | None = Field(
        None,
        description=(
            "A raw Eurostat dataset code for anything not covered by 'indicator', e.g. "
            "'nama_10_gdp'. Browse codes at https://ec.europa.eu/eurostat/databrowser. "
            "Provide either this or 'indicator', not both."
        ),
    )
    geo: str | None = Field(
        None,
        description="Eurostat geo code, e.g. 'DE' (Germany), 'FR' (France), 'EU27_2020' (the EU).",
    )
    since: str | None = Field(
        None,
        description="Only return observations from this period onward, e.g. '2020' or '2024-01'.",
    )
    filters: dict[str, str] | None = Field(
        None,
        description=(
            "Extra raw Eurostat dimension filters as code/value pairs, e.g. "
            "{'sex': 'T', 'unit': 'PC_ACT'}. Only needed for 'dataset_code' lookups where "
            "the dataset's dimensions aren't already covered by 'indicator' defaults."
        ),
    )


class EurostatTool(BaseTool):
    """Fetches live official EU statistics from Eurostat's public dissemination API.

    No API key or registration is required. Pass ``indicator`` for a well-known
    figure (unemployment_rate, inflation_rate, gdp, population), or
    ``dataset_code`` for any other raw Eurostat dataset.
    """

    BASE_URL: ClassVar[str] = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
    )
    REQUEST_TIMEOUT: ClassVar[int] = 30

    name: str = "Eurostat Statistics Fetcher"
    description: str = (
        "Fetches real, live official EU statistics from Eurostat. Pass 'indicator' for a "
        "well-known figure (unemployment_rate, inflation_rate, gdp, population) with a 'geo' "
        "code, or 'dataset_code' for any other raw Eurostat dataset. No API key required."
    )
    args_schema: type[BaseModel] = EurostatToolInput
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])
    env_vars: list[EnvVar] = Field(default_factory=list)

    def _run(
        self,
        indicator: str | None = None,
        dataset_code: str | None = None,
        geo: str | None = None,
        since: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> str:
        if not indicator and not dataset_code:
            known = ", ".join(KNOWN_INDICATORS.keys())
            return f"Error: provide either 'indicator' (one of: {known}) or 'dataset_code'."

        merged_filters: dict[str, str] = dict(filters or {})
        if indicator:
            info = KNOWN_INDICATORS.get(indicator)
            if info is None:
                known = ", ".join(KNOWN_INDICATORS.keys())
                return f"Error: unknown indicator '{indicator}'. Known indicators: {known}."
            dataset_code = info["dataset"]
            merged_filters = {**info["filters"], **merged_filters}

        if dataset_code is None:
            # Unreachable: the guard clause above already ensures indicator or
            # dataset_code is set, and the indicator branch always assigns one.
            return "Error: no dataset code resolved."

        try:
            data = self._fetch(dataset_code, geo, since, merged_filters)
            parsed = self._parse_jsonstat(data)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 404:
                return "Error: dataset code not found, or no data matches the given filters."
            if status == 400:
                return "Error: invalid dataset code or filter value. Check dimension codes and try again."
            if status == 429:
                return "Error: Eurostat rate limit exceeded. Try again later."
            return f"Error: Eurostat API error ({status})."
        except Exception as e:
            logger.exception(f"EurostatTool error fetching '{dataset_code}'")
            return f"Error: failed to fetch or parse Eurostat dataset '{dataset_code}': {e}"

        if not parsed["records"]:
            return f"No data found for dataset '{dataset_code}' with the given filters."

        return self._format_result(parsed)

    def _fetch(
        self,
        dataset_code: str,
        geo: str | None,
        since: str | None,
        filters: dict[str, str],
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"format": "JSON", "lang": "en", **filters}
        if geo:
            params["geo"] = geo
        if since:
            params["sinceTimePeriod"] = since

        logger.info(f"Fetching Eurostat dataset '{dataset_code}' with params {params}")
        response = requests.get(
            f"{self.BASE_URL}/{dataset_code}",
            params=params,
            timeout=self.REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())

    @staticmethod
    def _parse_jsonstat(data: dict[str, Any]) -> dict[str, Any]:
        """Unpack a Eurostat JSON-stat response into a flat list of records.

        JSON-stat encodes the result as a single flat `value` map keyed by a
        computed flat index, plus a `dimension` block describing the category
        labels for each axis. This walks the dimension sizes to recover, for
        every stored value, which category label it corresponds to on each axis.
        """
        dimension = data.get("dimension", {})
        ids: list[str] = data.get("id", [])
        sizes: list[int] = data.get("size", [])
        values: dict[str, float] = data.get("value", {})
        title = data.get("label", "")

        if not ids or not sizes or not values:
            return {"title": title, "records": []}

        categories: list[list[str]] = []
        for dim_id in ids:
            category = dimension.get(dim_id, {}).get("category", {})
            index = category.get("index", {})
            labels = category.get("label", {})
            ordered_codes: list[str | None] = [None] * len(index)
            for code, pos in index.items():
                ordered_codes[pos] = code
            categories.append(
                [labels.get(code, code) for code in ordered_codes if code is not None]
            )

        records: list[dict[str, Any]] = []
        for flat_index_str, value in values.items():
            remainder = int(flat_index_str)
            positions = [0] * len(sizes)
            for axis in range(len(sizes) - 1, -1, -1):
                positions[axis] = remainder % sizes[axis]
                remainder //= sizes[axis]
            record: dict[str, Any] = {
                dim_id: categories[axis][positions[axis]]
                for axis, dim_id in enumerate(ids)
            }
            record["value"] = value
            records.append(record)

        return {"title": title, "records": records}

    @staticmethod
    def _format_result(parsed: dict[str, Any]) -> str:
        lines = [parsed["title"]] if parsed["title"] else []
        for record in parsed["records"]:
            parts = [
                f"{key}={value}" for key, value in record.items() if key != "value"
            ]
            lines.append(f"{', '.join(parts)}: {record['value']}")
        return "\n".join(lines)
