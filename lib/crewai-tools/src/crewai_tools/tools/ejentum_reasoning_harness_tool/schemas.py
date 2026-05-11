from typing import Literal

from pydantic import BaseModel, Field


HarnessMode = Literal["reasoning", "code", "anti-deception", "memory"]


class EjentumHarnessParams(BaseModel):
    """Arguments for a single Ejentum harness call."""

    query: str = Field(
        ...,
        description=(
            "A 1-2 sentence description of the task the agent is about to work on. "
            "For mode='memory', format as: 'I noticed [X]. This might mean [Y]. "
            "Sharpen: [Z].'"
        ),
        min_length=1,
    )
    mode: HarnessMode = Field(
        ...,
        description=(
            "Which cognitive harness to retrieve a scaffold from. "
            "'reasoning' for analytical/diagnostic/planning/multi-step tasks. "
            "'code' for code generation, refactoring, review, debugging. "
            "'anti-deception' when the prompt pressures the agent to validate, "
            "certify, or soften an honest assessment. "
            "'memory' only when sharpening an observation already formed about "
            "cross-turn drift."
        ),
    )
