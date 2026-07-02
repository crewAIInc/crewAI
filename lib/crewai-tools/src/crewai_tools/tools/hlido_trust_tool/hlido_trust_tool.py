from typing import TYPE_CHECKING, Any, List, Optional, Type

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

if TYPE_CHECKING:
    from hlido_trust import HlidoClient


try:
    from hlido_trust import HlidoClient

    HLIDO_TRUST_AVAILABLE = True
except ImportError:
    HLIDO_TRUST_AVAILABLE = False


class HlidoTrustToolSchema(BaseModel):
    """Input for HlidoTrustTool."""

    slug: str = Field(
        ...,
        description="The Hlido agent slug to vet, e.g. 'aider' or 'crewai'.",
    )
    min_score: float = Field(
        70.0,
        description="Minimum acceptable Hlido score (0-100). 70 = STEADY tier or above.",
    )


class HlidoTrustTool(BaseTool):
    """Vet an AI agent against its independent Hlido review before delegating to it.

    `Hlido <https://hlido.eu>`_ publishes independent, evidence-backed trust scores
    for AI agents. This tool returns a PASS/FAIL gate plus the agent's score, tier,
    what it fails at, red flags, and an evidence URL — so an agent can decide whether
    to trust a tool or sub-agent before relying on it. Agents with no Hlido review, or
    with recorded red flags, fail the gate (fail-closed).

    No API key is required for trust checks. Set ``HLIDO_API_KEY`` (a ``hlk_live_*``
    key from https://hlido.eu/api/) only if you also need top-k recommendations.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, frozen=False
    )
    name: str = "Hlido Agent Trust Check"
    description: str = (
        "Vet an AI agent against its independent, evidence-backed Hlido review by slug "
        "before delegating to it. Returns a PASS/FAIL trust gate plus score, tier, red "
        "flags, and an evidence URL."
    )
    args_schema: Type[BaseModel] = HlidoTrustToolSchema
    api_key: Optional[str] = None
    _client: Optional["HlidoClient"] = PrivateAttr(None)
    package_dependencies: List[str] = ["hlido-trust"]
    env_vars: List[EnvVar] = [
        EnvVar(
            name="HLIDO_API_KEY",
            description=(
                "Optional Hlido API key (hlk_live_*) for top-k recommendations; "
                "not required for trust checks"
            ),
            required=False,
        ),
    ]

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        if not HLIDO_TRUST_AVAILABLE:
            raise ImportError(
                "`hlido-trust` package is required to use the HlidoTrustTool. "
                "Install it with `uv add hlido-trust` or `pip install hlido-trust`."
            )
        self.api_key = api_key
        self._client = HlidoClient(api_key=api_key, framework="crewai")

    def _run(self, **kwargs: Any) -> str:
        slug = kwargs.get("slug")
        if not slug:
            raise ValueError("`slug` is required to run a Hlido trust check")
        min_score = kwargs.get("min_score", 70.0)
        verdict = self._client.trust_check(slug)
        gate = "PASS" if verdict.recommended(min_score=min_score) else "FAIL"
        return f"[{gate} @ min_score={min_score}] {verdict.summary()}"
