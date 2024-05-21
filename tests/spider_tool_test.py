import os
from crewai_tools.tools.spider_tool.spider_tool import SpiderTool
from crewai import Agent, Task, Crew

def test_spider_tool():
    spider_tool = SpiderTool()
    
    searcher = Agent(
        role="Web Research Expert",
        goal="Find related information from specific URL's",
        backstory="An expert web researcher that uses the web extremely well",
        tools=[spider_tool],
        verbose=True
    )
    
    summarize_spider = Task(
        description="Summarize the content of spider.cloud",
        expected_output="A summary that goes over what spider does",
        agent=searcher
    )
    
    crew = Crew(
        agents=[searcher],
        tasks=[summarize_spider],
        verbose=2
    )
    
    crew.kickoff()

if __name__ == "__main__":
    test_spider_tool()