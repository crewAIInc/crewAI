from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task, crew

@CrewBase
class YourCrewName:
    """Description of your crew"""

    @agent
    def agent_one(self) -> Agent:
        return Agent(
            role="Test Agent",
            goal="Test Goal",
            backstory="Test Backstory",
            verbose=True
        )

    @task
    def task_one(self) -> Task:
        return Task(
            description="Test Description",
            expected_output="Test Output",
            agent=self.agent_one()
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.agent_one()],
            tasks=[self.task_one()],
            process=Process.sequential,
            verbose=True,
        )

c = YourCrewName()
result = c.kickoff()
print(result)
