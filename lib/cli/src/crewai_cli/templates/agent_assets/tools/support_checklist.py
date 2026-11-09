import json

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SupportChecklistInput(BaseModel):
    topic: str = Field(description="Troubleshooting topic: sign-in or billing.")


class SupportChecklistTool(BaseTool):
    name: str = "support_checklist"
    description: str = "Return a troubleshooting checklist for the fictional service."
    args_schema: type[BaseModel] = SupportChecklistInput

    def _run(self, topic: str) -> str:
        checklists = {
            "sign-in": [
                "Confirm the account email and whether the account uses SSO.",
                "Use the password-reset option or the organization's SSO sign-in.",
                "If the issue persists, share the error message with support without passwords or codes.",
            ],
            "billing": [
                "Check the invoice date and plan name.",
                "Compare the billed seats with the example plan allowance.",
                "Contact support with the invoice reference, never payment-card details.",
            ],
        }
        name = topic.strip().lower()
        if name not in checklists:
            return json.dumps(
                {"error": "Unknown topic", "available_topics": list(checklists)}
            )
        return json.dumps({"topic": name, "steps": checklists[name]})
