from typing import Tuple

from crewai.flows import Flow, end_job, router, start_job  # type: ignore


class RouterFlow(Flow):

    @start_job()
    @router()
    async def classify_email(self, report: str) -> Tuple[str, str]:
        if "urgent" in report:
            return "urgent", report

        return "normal", report

    @end_job("urgent")
    async def write_urgent_email(self, report: str) -> str:
        return f"Urgent Email Response: {report}"

    @end_job("normal")
    async def write_normal_email(self, report: str) -> str:
        return f"Normal Email Response: {report}"
