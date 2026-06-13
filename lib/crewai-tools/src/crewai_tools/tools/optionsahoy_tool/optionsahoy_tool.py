import json
from typing import Any, Literal

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


CalculatorName = Literal[
    "amt-iso",
    "nso",
    "rsu-sell-vs-hold",
    "concentration",
    "protective-put",
    "qsbs",
    "equity-funding",
]

# Required input fields per calculator, mirroring the published OpenAPI schema at
# https://optionsahoy.com/openapi.json. These are used for a friendly pre-flight
# check; the API remains the source of truth for validation.
REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "amt-iso": (
        "shares",
        "strike",
        "fmv",
        "filingStatus",
        "ordinaryIncome",
        "stateCode",
        "carryforwardCredit",
        "horizon",
        "cashReturnRate",
        "grantDate",
        "hasLeftCompany",
        "terminationDate",
    ),
    "nso": (
        "shares",
        "strike",
        "currentPrice",
        "ordinaryIncome",
        "filingStatus",
        "stateCode",
        "stillEmployed",
        "holdYears",
        "holdFunding",
    ),
    "rsu-sell-vs-hold": (
        "shares",
        "currentPrice",
        "ordinaryIncome",
        "filingStatus",
        "stateCode",
        "stillEmployed",
        "holdYears",
    ),
    "concentration": (
        "positionValue",
        "costBasis",
        "acquisitionDate",
        "sector",
        "stateCode",
        "filingStatus",
        "ordinaryIncome",
        "totalAssets",
    ),
    "protective-put": (
        "positionValue",
        "sector",
        "protectionLevel",
        "tenorYears",
    ),
    "qsbs": (
        "acquisitionDate",
        "saleDate",
        "entityType",
        "acquisitionMethod",
        "assetCategory",
        "industry",
        "activeBusiness",
        "adjustedBasis",
        "expectedGain",
        "stateCode",
        "ordinaryIncome",
        "filingStatus",
    ),
    "equity-funding": (
        "targetAfterTax",
        "targetDate",
        "ordinaryIncome",
        "filingStatus",
        "stateCode",
    ),
}

# ``terminationDate`` is meaningful when null (it encodes "no termination") so it
# must be sent even when the caller leaves it unset.
_KEEP_NULL = {"terminationDate"}


class OptionsAhoyToolSchema(BaseModel):
    """Input for OptionsAhoyTool."""

    calculator: CalculatorName = Field(
        ...,
        description=(
            "Which equity-compensation calculator to run. One of: "
            "'amt-iso' (optimize a multi-year incentive stock option (ISO) exercise "
            "schedule under the alternative minimum tax (AMT)); "
            "'nso' (tax and after-tax proceeds of exercising non-qualified stock "
            "options (NSOs), holding versus selling); "
            "'rsu-sell-vs-hold' (sell vested restricted stock units (RSUs) at vest "
            "versus hold, on an after-tax, risk-adjusted basis); "
            "'concentration' (analyze a concentrated single-stock position and the "
            "after-tax cost of diversifying it); "
            "'protective-put' (price a protective put hedge at a given downside "
            "protection level and tenor); "
            "'qsbs' (check qualified small business stock (QSBS) eligibility and the "
            "resulting capital-gains exclusion); "
            "'equity-funding' (plan which equity lots to sell, and when, to fund a "
            "cash goal by a target date at the least after-tax cost)."
        ),
    )
    inputs: dict[str, Any] = Field(
        ...,
        description=(
            "The calculator inputs as a JSON object. Field names match the OptionsAhoy "
            "public schema exactly (for example: shares, strike, fmv, filingStatus, "
            "ordinaryIncome, stateCode, grantDate). Money values are plain numbers, "
            "dates are ISO strings (YYYY-MM-DD), and US state codes are two letters."
        ),
    )


class OptionsAhoyTool(BaseTool):
    """Run an OptionsAhoy equity-compensation tax calculator.

    OptionsAhoy is a keyless public REST API that computes the tax outcome of common
    equity-compensation decisions against the federal tax code and all fifty states
    plus the District of Columbia. This tool wraps the seven calculators behind a
    single ``calculator`` selector plus an ``inputs`` object. No API key is read,
    stored, or sent.
    """

    name: str = "OptionsAhoy Equity Compensation Tax Calculator"
    description: str = (
        "Computes the tax outcome of equity-compensation decisions using the "
        "OptionsAhoy public API. Choose a 'calculator' and pass its 'inputs': "
        "incentive stock option (ISO) exercise under the alternative minimum tax "
        "(AMT), non-qualified stock option (NSO) exercise, restricted stock unit "
        "(RSU) sell-versus-hold, single-stock concentration, protective put pricing, "
        "qualified small business stock (QSBS) eligibility, and funding a cash goal "
        "from equity lots. Returns the calculator's JSON result. No API key required."
    )
    args_schema: type[BaseModel] = OptionsAhoyToolSchema
    base_url: str = "https://optionsahoy.com"
    timeout: int = 30

    def _check_required(self, calculator: str, inputs: dict[str, Any]) -> None:
        required = REQUIRED_FIELDS.get(calculator, ())
        missing = [field for field in required if field not in inputs]
        if missing:
            raise ValueError(
                f"OptionsAhoy '{calculator}' is missing required input field(s): "
                f"{', '.join(missing)}"
            )

    def _build_payload(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in inputs.items()
            if value is not None or key in _KEEP_NULL
        }

    def _run(self, calculator: str, inputs: dict[str, Any]) -> str:
        """Call the selected OptionsAhoy calculator and return its JSON result.

        Args:
            calculator: The calculator endpoint to run.
            inputs: The calculator inputs, matching the OptionsAhoy public schema.

        Returns:
            A JSON string with the calculator result, or a JSON error object when the
            request fails.
        """
        self._check_required(calculator, inputs)
        url = f"{self.base_url.rstrip('/')}/api/v1/{calculator}"
        payload = self._build_payload(inputs)

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            detail: Any
            try:
                detail = exc.response.json()
            except ValueError:
                detail = exc.response.text
            message = (
                f"OptionsAhoy '{calculator}' request failed "
                f"({exc.response.status_code})"
            )
            if isinstance(detail, dict) and detail.get("error"):
                message = f"{message}: {detail['error']}"
            return json.dumps({"error": message, "detail": detail})
        except requests.exceptions.RequestException as exc:
            return json.dumps(
                {"error": f"OptionsAhoy '{calculator}' request failed: {exc}"}
            )

        try:
            result = response.json()
        except ValueError:
            return json.dumps(
                {"error": (f"OptionsAhoy '{calculator}' returned a non-JSON response")}
            )

        return json.dumps(result, indent=2)
