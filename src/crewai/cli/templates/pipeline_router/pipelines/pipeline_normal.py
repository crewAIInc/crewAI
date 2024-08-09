from crewai import Pipeline
from crewai.project import PipelineBase
from ..crews.normal_crew.normal_crew import NormalCrew


@PipelineBase
class NormalPipeline:
    def __init__(self):
        # Initialize crews
        self.normal_crew = NormalCrew().crew()

    def create_pipeline(self):
        return Pipeline(
            stages=[
                self.normal_crew
            ]
        )
    
    async def kickoff(self, inputs):
        pipeline = self.create_pipeline()
        results = await pipeline.kickoff(inputs)
        return results


