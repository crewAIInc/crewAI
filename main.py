from src.agent import CryptoResearchAgent
from crewai import Crew, Process, Task

class CryptoResearchCrew:
    def __init__(self, project_name):
        self.project_name = project_name
        self.agent = CryptoResearchAgent()

    def run(self):
        project_analyzer_agent = self.agent.project_analyzer_agent()
        community_analyzer_agent = self.agent.community_analyzer_agent()
        funds_analyzer_agent = self.agent.funds_analyzer_agent()
        collaborations_analyzer_agent = self.agent.collaborations_analyzer_agent()

        project_analysis_task = Task(
            description=f"Analyze the {self.project_name} project.",
            expected_output="A detailed analysis of the project's technology, team, and roadmap.",
            agent=project_analyzer_agent,
        )

        community_analysis_task = Task(
            description=f"Analyze the {self.project_name} community.",
            expected_output="A detailed analysis of the project's community engagement, social media presence, and overall sentiment.",
            agent=community_analyzer_agent,
        )

        funds_analysis_task = Task(
            description=f"Analyze the {self.project_name} funds.",
            expected_output="A detailed analysis of the project's funding, tokenomics, and overall financial health.",
            agent=funds_analyzer_agent,
        )

        collaborations_analysis_task = Task(
            description=f"Analyze the {self.project_name} collaborations.",
            expected_output="A detailed analysis of the project's partnerships, collaborations, and alliances.",
            agent=collaborations_analyzer_agent,
        )

        crew = Crew(
            agents=[
                project_analyzer_agent,
                community_analyzer_agent,
                funds_analyzer_agent,
                collaborations_analyzer_agent,
            ],
            tasks=[
                project_analysis_task,
                community_analysis_task,
                funds_analysis_task,
                collaborations_analysis_task,
            ],
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
    project_name = input("Enter the crypto project name to analyze: ")
    crypto_research_crew = CryptoResearchCrew(project_name)
    result = crypto_research_crew.run()
    print(result)
