from crewai.tools import BaseTool


class SupportHoursTool(BaseTool):
    name: str = "support_hours"
    description: str = "Return the fictional service's published support hours."

    def _run(self) -> str:
        return "Support is available Monday to Friday, 09:00-17:00 UTC."
