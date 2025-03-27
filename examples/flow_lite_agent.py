from typing import List, cast

from crewai_tools.tools.website_search.website_search_tool import WebsiteSearchTool
from pydantic import BaseModel, Field

from crewai.flow.flow import Flow, listen, start
from crewai.lite_agent import LiteAgent


# Define a structured output format
class MarketAnalysis(BaseModel):
    key_trends: List[str] = Field(description="List of identified market trends")
    market_size: str = Field(description="Estimated market size")
    competitors: List[str] = Field(description="Major competitors in the space")


# Define flow state
class MarketResearchState(BaseModel):
    product: str = ""
    analysis: MarketAnalysis | None = None


# Create a flow class
class MarketResearchFlow(Flow[MarketResearchState]):
    @start()
    def initialize_research(self):
        print(f"Starting market research for {self.state.product}")

    @listen(initialize_research)
    def analyze_market(self):
        # Create a LiteAgent for market research
        analyst = LiteAgent(
            role="Market Research Analyst",
            goal=f"Analyze the market for {self.state.product}",
            backstory="You are an experienced market analyst with expertise in "
            "identifying market trends and opportunities.",
            llm="gpt-4o",
            tools=[WebsiteSearchTool()],
            verbose=True,
            response_format=MarketAnalysis,
        )

        # Define the research query
        query = f"""
        Research the market for {self.state.product}. Include:
        1. Key market trends
        2. Market size
        3. Major competitors
        
        Format your response according to the specified structure.
        """

        # Execute the analysis
        result = analyst.kickoff(query)
        self.state.analysis = cast(MarketAnalysis, result.pydantic)
        return result.pydantic

    @listen(analyze_market)
    def present_results(self):
        analysis = self.state.analysis
        if analysis is None:
            print("No analysis results available")
            return

        print("\nMarket Analysis Results")
        print("=====================")

        print("\nKey Market Trends:")
        for trend in analysis.key_trends:
            print(f"- {trend}")

        print(f"\nMarket Size: {analysis.market_size}")

        print("\nMajor Competitors:")
        for competitor in analysis.competitors:
            print(f"- {competitor}")


# Usage example
flow = MarketResearchFlow()
result = flow.kickoff(inputs={"product": "AI-powered chatbots"})
