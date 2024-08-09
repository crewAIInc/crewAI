from crewai import Pipeline
from crewai.project import PipelineBase
from ..crews.urgent_crew.urgent_crew import UrgentCrew

@PipelineBase
class UrgentPipeline:
    def __init__(self):
        # Initialize crews
        self.urgent_crew = UrgentCrew().crew()
    
    def create_pipeline(self):
        return Pipeline(
            stages=[
                self.urgent_crew
            ]
        )
    
    async def kickoff(self, inputs):
        pipeline = self.create_pipeline()
        results = await pipeline.kickoff(inputs)
        return results


