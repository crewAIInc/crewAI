"""
Content Writing Crew
A multi-agent system for creating high-quality blog articles and content.
"""

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List


@CrewBase
class ContentWritingCrew:
    """Content Writing Crew for creating professional blog articles"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # Agent Definitions
    @agent
    def content_planner(self) -> Agent:
        """
        Content Strategy Expert
        Creates comprehensive content plans and outlines
        """
        return Agent(
            config=self.agents_config['content_planner'],
            verbose=True
        )

    @agent
    def researcher(self) -> Agent:
        """
        Expert Research Analyst
        Conducts thorough research and gathers credible information
        """
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True,
            # Could add research tools here:
            # tools=[SerperDevTool(), FileReadTool()]
        )

    @agent
    def content_writer(self) -> Agent:
        """
        Professional Content Writer
        Writes engaging, well-structured content
        """
        return Agent(
            config=self.agents_config['content_writer'],
            verbose=True
        )

    @agent
    def seo_specialist(self) -> Agent:
        """
        SEO Optimization Expert
        Optimizes content for search engines
        """
        return Agent(
            config=self.agents_config['seo_specialist'],
            verbose=True
        )

    @agent
    def editor(self) -> Agent:
        """
        Senior Content Editor
        Reviews and polishes content for publication
        """
        return Agent(
            config=self.agents_config['editor'],
            verbose=True
        )

    # Task Definitions
    @task
    def plan_content(self) -> Task:
        """Create comprehensive content plan and outline"""
        return Task(
            config=self.tasks_config['plan_content'],
            agent=self.content_planner()
        )

    @task
    def research_topic(self) -> Task:
        """Conduct thorough research on the topic"""
        return Task(
            config=self.tasks_config['research_topic'],
            agent=self.researcher()
        )

    @task
    def write_article(self) -> Task:
        """Write the complete article"""
        return Task(
            config=self.tasks_config['write_article'],
            agent=self.content_writer(),
            output_file='output/draft_article.md'
        )

    @task
    def optimize_seo(self) -> Task:
        """Optimize article for SEO"""
        return Task(
            config=self.tasks_config['optimize_seo'],
            agent=self.seo_specialist()
        )

    @task
    def edit_and_polish(self) -> Task:
        """Final editing and polishing"""
        return Task(
            config=self.tasks_config['edit_and_polish'],
            agent=self.editor(),
            output_file='output/final_article.md'
        )

    @crew
    def crew(self) -> Crew:
        """
        Creates the Content Writing Crew

        Returns:
            Crew: A crew with 5 agents working sequentially to create content
        """
        return Crew(
            agents=self.agents,  # Automatically uses all @agent decorated methods
            tasks=self.tasks,    # Automatically uses all @task decorated methods
            process=Process.sequential,  # Tasks run in order
            verbose=True,
            memory=True,  # Enable crew memory for context retention
            # Optional: Add embedder for better memory
            # embedder={
            #     "provider": "openai",
            #     "config": {"model": "text-embedding-3-small"}
            # }
        )


def run():
    """
    Run the Content Writing Crew with default inputs
    """
    inputs = {
        'topic': 'AI-Powered Content Creation Tools',
        'content_goal': 'Educate readers about AI content tools and provide actionable recommendations',
        'word_count': 2000
    }

    result = ContentWritingCrew().crew().kickoff(inputs=inputs)
    return result


if __name__ == "__main__":
    print("🚀 Starting Content Writing Crew...")
    print("=" * 60)
    result = run()
    print("=" * 60)
    print("✅ Content Writing Crew completed!")
    print(f"📄 Final output:\n{result}")
