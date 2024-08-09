from crewai import Pipeline
from crewai.project import PipelineBase
from ..crews.classifier_crew.classifier_crew import ClassifierCrew


@PipelineBase
class EmailClassifierPipeline:
    def __init__(self):
        # Initialize crews
        self.classifier_crew = ClassifierCrew().crew()

    def create_pipeline(self):
        return Pipeline(
            stages=[
                self.classifier_crew
            ]
        )
    
    async def kickoff(self, inputs):
        pipeline = self.create_pipeline()
        results = await pipeline.kickoff(inputs)
        return results


