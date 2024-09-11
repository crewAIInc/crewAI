from crewai.flows import Flow, end_job, start_job  # type: ignore


class SimpleFlow(Flow):

    @start_job()
    async def research(self, topic: str) -> str:
        print(f"Researching {topic}...")
        return f"Full report on {topic}..."

    @end_job("research")
    async def write_post(self, report: str) -> str:
        return f"Post written: {report}"
