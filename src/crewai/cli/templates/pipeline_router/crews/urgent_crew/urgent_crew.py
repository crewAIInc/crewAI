from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# Uncomment the following line to use an example of a custom tool
# from demo_pipeline.tools.custom_tool import MyCustomTool

# Check our tools documentations for more information on how to use them
# from crewai_tools import SerperDevTool


@CrewBase
class UrgentCrew:
    """Urgent Email Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def urgent_handler(self) -> Agent:
        return Agent(config=self.agents_config["urgent_handler"], verbose=True)

    @task
    def urgent_task(self) -> Task:
        return Task(
            config=self.tasks_config["urgent_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Urgent Email Crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
