from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel

# Uncomment the following line to use an example of a custom tool
# from demo_pipeline.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool

class UrgencyScore(BaseModel):
    urgency_score: int

@CrewBase
class ClassifierCrew:
    """Email Classifier Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def classifier(self) -> Agent:
        return Agent(config=self.agents_config["classifier"], verbose=True)

    @task
    def urgent_task(self) -> Task:
        return Task(
            config=self.tasks_config["classify_email"],
            output_pydantic=UrgencyScore,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Email Classifier Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
