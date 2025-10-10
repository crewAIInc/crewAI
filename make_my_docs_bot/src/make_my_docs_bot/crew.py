from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from make_my_docs_bot.tools.branch_change_analyzer import BranchChangeAnalyzerTool
from make_my_docs_bot.tools.korean_reverse_mapping import KoreanFileMappingTool
from make_my_docs_bot.tools.portuguese_brazil_reverse_mapping import PortugureseBrazilFileMappingTool
from make_my_docs_bot.tools.read_write_tool import FileContentUpdaterTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class MakeMyDocsBot():
    """MakeMyDocsBot crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def documentation_change_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['documentation_change_analyzer'], # type: ignore[index]
            verbose=False,
            max_iter=2,
            tools=[
				BranchChangeAnalyzerTool(result_as_answer=True)
            ]
        )
    

    @agent
    def korean_translator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['korean_translator_agent'], # type: ignore[index]
            verbose=False,
            max_iter=2,
            max_retry_limit=1,
            tools=[
				KoreanFileMappingTool()
            ]
        )
    
    @agent
    def portuguese_brazil_translator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['portuguese_brazil_translator_agent'], # type: ignore[index]
            verbose=False,
            max_iter=2,
            tools=[
				PortugureseBrazilFileMappingTool()
            ]
        )
    
    @agent
    def content_update_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['content_update_agent'], # type: ignore[index]
            verbose=False,
            max_iter=2,
            tools=[
				FileContentUpdaterTool()
            ]
        )
    # @agent
    # def reporting_analyst(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['reporting_analyst'], # type: ignore[index]
    #         verbose=True
    #     )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def analyze_branch_documentation_changes_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_branch_documentation_changes_task'], # type: ignore[index]
        )
    
    @task
    def korean_documentation_translator_task(self) -> Task:
        return Task(
            config=self.tasks_config['korean_documentation_translator_task'], # type: ignore[index]
        )
    
    @task
    def portuguese_brazil_documentation_translator_task(self) -> Task:
        return Task(
            config=self.tasks_config['portuguese_brazil_documentation_translator_task'], # type: ignore[index]
        )
    
    @task
    def content_update_task(self) -> Task:
        return Task(
            config=self.tasks_config['content_update_task'], # type: ignore[index]
        )


    # @task
    # def reporting_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['reporting_task'], # type: ignore[index]
    #         output_file='report.md'
    #     )

    @crew
    def crew(self) -> Crew:
        """Creates the MakeMyDocsBot crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            tracing=True
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
