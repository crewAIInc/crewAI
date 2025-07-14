from crewai import Agent
from crewai_tools import SerperDevTool

class CryptoResearchAgent:
    def __init__(self):
        self.search_tool = SerperDevTool()

    def project_analyzer_agent(self):
        return Agent(
            role="Project Analyzer",
            goal="Analyze the crypto project's technology, team, and roadmap.",
            backstory="An expert in blockchain technology and software development, you can analyze the technical aspects of a crypto project.",
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=True,
        )

    def community_analyzer_agent(self):
        return Agent(
            role="Community Analyzer",
            goal="Analyze the crypto project's community engagement, social media presence, and overall sentiment.",
            backstory="A social media expert, you can analyze the community's sentiment and engagement.",
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=True,
        )

    def funds_analyzer_agent(self):
        return Agent(
            role="Funds Analyzer",
            goal="Analyze the crypto project's funding, tokenomics, and overall financial health.",
            backstory="A financial expert, you can analyze the financial aspects of a crypto project.",
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=True,
        )

    def collaborations_analyzer_agent(self):
        return Agent(
            role="Collaborations Analyzer",
            goal="Analyze the crypto project's partnerships, collaborations, and alliances.",
            backstory="A networking expert, you can analyze the partnerships and collaborations of a crypto project.",
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=True,
        )
