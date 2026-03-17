from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from crewai.project import CrewBase, agent, crew, task, llm
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool

from crew_trader_pro.app.schemas.chief_strategist import ChiefStrategyOutput

@CrewBase
class CrewTraderPro():
    """CrewTraderPro crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @llm
    def deepseek_llm(self) -> LLM:
        import os
        return LLM(
            model="deepseek/deepseek-chat",
            api_base="https://api.deepseek.com/v1",
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            temperature=0.15,
        )

    # ── Agent 1: K线形态分析师 ──
    @agent
    def kline_pattern_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['kline_pattern_analyst'],  # type: ignore[index]
            verbose=True
        )

    # ── Agent 2: 技术指标分析师 ──
    @agent
    def technical_indicator_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_indicator_analyst'],  # type: ignore[index]
            verbose=True
        )

    # ── Agent 3: 市场情绪分析师（暂时禁用）──
    # @agent
    # def sentiment_analyst(self) -> Agent:
    #     return Agent(
    #         config=self.agents_config['sentiment_analyst'],
    #         tools=[SerperDevTool()],
    #         verbose=True
    #     )

    # ── Agent 4: 首席交易策略师 ──
    @agent
    def chief_strategist(self) -> Agent:
        return Agent(
            config=self.agents_config['chief_strategist'],  # type: ignore[index]
            verbose=True
        )

    # ── Task 1: K线形态分析 ──
    @task
    def kline_pattern_task(self) -> Task:
        return Task(
            config=self.tasks_config['kline_pattern_task'],  # type: ignore[index]
        )

    # ── Task 2: 技术指标分析 ──
    @task
    def technical_indicator_task(self) -> Task:
        return Task(
            config=self.tasks_config['technical_indicator_task'],  # type: ignore[index]
        )

    # ── Task 3: 市场情绪分析（暂时禁用）──
    # @task
    # def sentiment_analysis_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['sentiment_analysis_task'],
    #     )

    # ── Task 4（暂为Task 3）: 首席策略师综合决策 ──
    @task
    def chief_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['chief_strategy_task'],  # type: ignore[index]
            output_json=ChiefStrategyOutput,
            output_file='trading_decision_report.json'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewTraderPro crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,  # 顺序执行：1→2→3→4，前一个输出自动传给下一个
            verbose=True,
        )
