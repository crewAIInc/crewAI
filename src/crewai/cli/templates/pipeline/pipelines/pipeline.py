"""
This pipeline file includes two different examples to demonstrate the flexibility of crewAI pipelines.

Example 1: Two-Stage Pipeline
-----------------------------
This pipeline consists of two crews:
1. ResearchCrew: Performs research on a given topic.
2. WriteXCrew: Generates an X (Twitter) post based on the research findings.

Key features:
- The ResearchCrew's final task uses output_json to store all research findings in a JSON object.
- This JSON object is then passed to the WriteXCrew, where tasks can access the research findings.

Example 2: Two-Stage Pipeline with Parallel Execution
-------------------------------------------------------
This pipeline consists of three crews:
1. ResearchCrew: Performs research on a given topic.
2. WriteXCrew and WriteLinkedInCrew: Run in parallel, using the research findings to generate posts for X and LinkedIn, respectively.

Key features:
- Demonstrates the ability to run multiple crews in parallel.
- Shows how to structure a pipeline with both sequential and parallel stages.

Usage:
- To switch between examples, comment/uncomment the respective code blocks below.
- Ensure that you have implemented all necessary crew classes (ResearchCrew, WriteXCrew, WriteLinkedInCrew) before running.
"""

# Common imports for both examples
from crewai import Pipeline



# Uncomment the crews you need for your chosen example
from ..crews.research_crew.research_crew import ResearchCrew
from ..crews.write_x_crew.write_x_crew import WriteXCrew
# from .crews.write_linkedin_crew.write_linkedin_crew import WriteLinkedInCrew  # Uncomment for Example 2

# EXAMPLE 1: Two-Stage Pipeline
# -----------------------------
# Uncomment the following code block to use Example 1

class {{pipeline_name}}Pipeline:
    def __init__(self):
        # Initialize crews
        self.research_crew = ResearchCrew().crew()
        self.write_x_crew = WriteXCrew().crew()
    
    def create_pipeline(self):
        return Pipeline(
            stages=[
                self.research_crew,
                self.write_x_crew
            ]
        )
    
    async def kickoff(self, inputs):
        pipeline = self.create_pipeline()
        results = await pipeline.kickoff(inputs)
        return results


# EXAMPLE 2: Two-Stage Pipeline with Parallel Execution
# -------------------------------------------------------
# Uncomment the following code block to use Example 2

# @PipelineBase
# class {{pipeline_name}}Pipeline:
#     def __init__(self):
#         # Initialize crews
#         self.research_crew = ResearchCrew().crew()
#         self.write_x_crew = WriteXCrew().crew()
#         self.write_linkedin_crew = WriteLinkedInCrew().crew()
    
#     @pipeline
#     def create_pipeline(self):
#         return Pipeline(
#             stages=[
#                 self.research_crew,
#                 [self.write_x_crew, self.write_linkedin_crew]  # Parallel execution
#             ]
#         )

#     async def run(self, inputs):
#         pipeline = self.create_pipeline()
#         results = await pipeline.kickoff(inputs)
#         return results