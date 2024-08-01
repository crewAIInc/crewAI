from crewai import Pipeline
from crewai.project import PipelineBase
from crews.crew import *

@PipelineBase
class {{pipeline_name}}Pipeline:
    def __init__(self):
        # Initialize crews
        {% for crew_name in crew_names %}
        self.{{crew_name.lower()}}_crew = {{crew_name}}Crew().crew()
        {% endfor %}

    @pipeline
    def create_pipeline(self):
        return Pipeline(
            stages=[
                {% for crew_name in crew_names %}
                self.{{crew_name.lower()}}_crew,
                {% endfor %}
            ]
        )

    async def run(self, inputs):
        pipeline = self.create_pipeline()
        results = await pipeline.kickoff(inputs)
        return results