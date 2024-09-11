from crewai.flows import Flow, end_job, job, start_job  # type: ignore


class SimpleFlow(Flow):

    @start_job()
    async def research_crew(self, topic: str) -> str:
        result = research_crew.kickoff(inputs={topic: topic})
        return result.raw

    @job("research_crew")
    async def create_x_post(self, research: str) -> str:
        result = x_post_crew.kickoff(inputs={research: research})
        return result.raw

    @end_job("research")
    async def post_to_x(self, post: str) -> None:
        # TODO: Post to X
        return None
