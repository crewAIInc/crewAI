from pydantic import BaseModel

from crewai import Agent, Crew, Task
from crewai.flow import Flow, start, listen

class AnalysisState(BaseModel):
    topic: str = ""
    research_results: str = ""
    analysis: str = ""

class AnalysisFlow(Flow[AnalysisState]):
    def __init__(self):
        super().__init__()
        
        # Create agents
        self.researcher = Agent(
            role="Researcher",
            goal="Research the latest information",
            backstory="You are an expert researcher"
        )
        
        self.analyst = Agent(
            role="Analyst",
            goal="Analyze research findings",
            backstory="You are an expert analyst"
        )
        
    @start()
    def start_research(self):
        print(f"Starting research on topic: {self.state.topic}")
        
        # Create research task
        research_task = Task(
            description=f"Research the latest information about {self.state.topic}",
            expected_output="A summary of research findings",
            agent=self.researcher
        )
        
        # Run research task
        crew = Crew(agents=[self.researcher], tasks=[research_task])
        result = crew.kickoff()
        
        self.state.research_results = result.raw
        return result.raw
        
    @listen(start_research)
    def analyze_results(self, research_results):
        print("Analyzing research results")
        
        # Create analysis task
        analysis_task = Task(
            description=f"Analyze the following research results: {research_results}",
            expected_output="A detailed analysis",
            agent=self.analyst
        )
        
        # Run analysis task
        crew = Crew(agents=[self.analyst], tasks=[analysis_task])
        result = crew.kickoff()
        
        self.state.analysis = result.raw
        return result.raw

# For testing
if __name__ == "__main__":
    flow = AnalysisFlow()
    result = flow.kickoff(inputs={"topic": "Artificial Intelligence"})
    print(f"Final result: {result}")
