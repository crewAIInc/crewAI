import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class LookupPlanInput(BaseModel):
    plan: str = Field(description="Example plan name: starter or pro.")


class LookupPlanTool(BaseTool):
    name: str = "lookup_plan"
    description: str = "Look up the fictional service's example plan features."
    args_schema: type[BaseModel] = LookupPlanInput

    def _run(self, plan: str) -> str:
        plans = {
            "starter": {"seats": 1, "support": "email"},
            "pro": {"seats": 5, "support": "priority email"},
        }
        name = plan.strip().lower()
        if name not in plans:
            return json.dumps(
                {"error": "Unknown example plan", "available_plans": list(plans)}
            )
        return json.dumps({"plan": name, **plans[name]})
