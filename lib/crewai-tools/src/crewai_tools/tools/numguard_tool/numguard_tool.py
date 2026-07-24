import json
import logging
from typing import List, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field

logger = logging.getLogger(__file__)

_EXAMPLES = (
    "backtest -> {'sr': 0.12, 'T': 250, 'n_trials': 100}; "
    "model_gap -> {'n': 2000, 'p1': 0.85, 'p2': 0.80}; "
    "subset_win -> {'p': 0.03, 'n_tests': 20}; "
    "judge_bias -> {'wins': 68, 'n': 100}"
)


class NumGuardToolInput(BaseModel):
    kind: str = Field(
        ...,
        description="The claim to check: 'backtest' (Deflated Sharpe Ratio), 'model_gap' (accuracy gap "
        "power), 'subset_win' (multiple-testing correction), or 'judge_bias' (LLM-judge preference).",
    )
    params: dict = Field(
        ...,
        description=f"Inputs for the check. Examples: {_EXAMPLES}.",
    )


class NumGuardTool(BaseTool):
    """Verify a number before an agent asserts it.

    Routes a numeric claim to the statistical check that most directly tests it and returns whether it survives
    plus an honest verdict — so an agent can gate a backtest Sharpe, an A/B accuracy gap, a cherry-picked subset
    win, or an LLM-judge preference before reporting it as real. Backed by the open-source ``numguard`` library
    (Deflated Sharpe Ratio + eval-integrity statistics).
    """

    name: str = "NumGuard Number Verifier"
    description: str = (
        "Verify a statistical claim before asserting it. Give it a 'kind' (backtest / model_gap / subset_win / "
        "judge_bias) and 'params'; it returns whether the number survives its check and the honest verdict to "
        "report instead. Use it before you state a backtest Sharpe, an A/B result, a subset win, or a judge "
        "preference as a fact."
    )
    args_schema: Type[BaseModel] = NumGuardToolInput
    package_dependencies: List[str] = ["numguard"]
    env_vars: List[EnvVar] = []

    def _run(self, kind: str, params: dict) -> str:
        try:
            from numguard.guard import check
        except ImportError:
            return "numguard is not installed. Run: pip install numguard"
        try:
            verdict = check(kind, **(params or {}))
        except Exception as e:
            logger.error(f"NumGuardTool error: {e}")
            return f"Could not verify claim '{kind}': {e}"
        survives = bool(verdict.get("survives"))
        line = verdict.get("verdict") or ("survives" if survives else "flagged")
        return json.dumps({"survives": survives, "verdict": line, "detail": verdict}, default=str)
