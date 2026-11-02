from typing import Any

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class GoldenDirectiveToolSchema(BaseModel):
    """Input for GoldenDirectiveTool."""

    agent_score: float = Field(
        ...,
        description=(
            "Mandatory agent score A: the agent's net contribution-health "
            "score, floored at 0, against a healthy baseline."
        ),
    )
    environment_score: float = Field(
        ...,
        description=(
            "Mandatory environment score E (full frame): the health of "
            "everything not-A, against a healthy baseline."
        ),
    )
    environment_score_cropped: float | None = Field(
        default=None,
        description=(
            "Optional environment score E computed on a cropped frame "
            "(a small measured slice of the environment). When given, "
            "the tool also reports cropped Y and the full-versus-cropped gap."
        ),
    )


def relationship_score(A: float, E: float) -> float:
    """Y = 2(A x E) / (A + E), with A and E floored at 0.

    If either A or E reaches 0, Y reaches 0.
    """
    A = max(0.0, float(A))
    E = max(0.0, float(E))
    if A == 0.0 or E == 0.0:
        return 0.0
    return (2.0 * (A * E)) / (A + E)


class GoldenDirectiveTool(BaseTool):
    """Compute the Golden Directive relationship score Y.

    The Golden Directive (https://thegoldendirective.pages.dev/) is a
    candidate training reward, proposed not deployed:

        Y = 2(A x E) / (A + E)

    A is an agent's net contribution-health score, designed to be
    grounded in resource ledgers (compute, energy, water, hardware,
    economics, and CO2 later), not self-report. E is the health of
    everything not-A, against a healthy baseline. This tool performs
    the calculation and the full-versus-cropped reporting only; it
    does not measure A or E, and A and E measurement is not yet
    operational for real training in the source proposal.
    """

    name: str = "Golden Directive relationship score"
    description: str = (
        "Calculates the Golden Directive relationship score "
        "Y = 2(A x E) / (A + E) from an agent score A and an environment "
        "health score E. If either A or E reaches 0, Y reaches 0. "
        "Optionally also reports Y on a cropped environment frame and "
        "the full-versus-cropped gap."
    )
    args_schema: type[BaseModel] = GoldenDirectiveToolSchema

    def _run(
        self,
        agent_score: float,
        environment_score: float,
        environment_score_cropped: float | None = None,
        **kwargs: Any,
    ) -> Any:
        A = max(0.0, float(agent_score))
        E = max(0.0, float(environment_score))
        y_full = relationship_score(A, E)
        lines = [
            f"A = {A}",
            f"E (full) = {E}",
            f"Y (full) = {y_full:.4f}",
        ]
        if environment_score_cropped is not None:
            E_cropped = max(0.0, float(environment_score_cropped))
            y_cropped = relationship_score(A, E_cropped)
            lines.append(f"E (cropped) = {E_cropped}")
            lines.append(f"Y (cropped) = {y_cropped:.4f}")
            lines.append(f"Gap (cropped Y - full Y) = {y_cropped - y_full:.4f}")
        return "\n".join(lines)
