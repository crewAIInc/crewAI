from crewai.flows import Flow, end_job, job, start_job  # type: ignore


class LongerFlow(Flow):

    @start_job()
    async def research(self, topic: str) -> str:
        print(f"Researching {topic}...")
        return f"Full report on {topic}..."

    @job("research")
    async def edit_report(self, report: str) -> str:
        return f"Edited report: {report}"

    @end_job("edit_report")
    async def write_post(self, report: str) -> str:
        return f"Post written: {report}"
