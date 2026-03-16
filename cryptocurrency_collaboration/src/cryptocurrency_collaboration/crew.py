from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, llm
from crewai.agents.agent_builder.base_agent import BaseAgent
from cryptocurrency_collaboration.tools.kline_tool import KlineTool
from cryptocurrency_collaboration.tools.sentiment_tool import SentimentTool
from crewai.llm import LLM

@CrewBase
class CryptocurrencyCollaboration():
    agents: list[BaseAgent]
    tasks: list[Task]

    # 实例化工具对象
    kline_tool = KlineTool()
    sentiment_tool = SentimentTool()

    @llm
    def local_gemma_llm(self) -> LLM:
        return LLM(
            model="ollama/gemma3:4b",
            api_base="http://localhost:11434",
        )

    @llm
    def deepseek_llm(self) -> LLM:
        import os
        return LLM(
            model="deepseek/deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
        )

    @agent
    def kline_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config['kline_analyzer'], # type: ignore[index]
            tools=[self.kline_tool],
            verbose=True
        )

    @agent
    def news_sentiment_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['news_sentiment_analyst'], # type: ignore[index]
            tools=[self.sentiment_tool],
            verbose=True
        )

    @agent
    def chief_decision_maker(self) -> Agent:
        return Agent(
            config=self.agents_config['chief_decision_maker'], # type: ignore[index]
            verbose=True
        )

    @task
    def research_kline(self) -> Task:
        return Task(
            config=self.tasks_config['research_kline'], # type: ignore[index]
        )

    @task
    def sentiment_news(self) -> Task:
        return Task(
            config=self.tasks_config['sentiment_news'], # type: ignore[index]
        )

    @task
    def decision_report(self) -> Task:
        return Task(
            config=self.tasks_config['decision_report'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
