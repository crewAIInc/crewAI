import os
from crewai_tools.tools.spider_full_tool.spider_full_tool import SpiderFullTool, SpiderFullParams
from crewai import Agent, Task, Crew

def test_spider_tool():
    spider_tool = SpiderFullTool()

    params = SpiderFullParams(
        return_format="markdown"
    )

    docs = spider_tool._run("https://spider.cloud", params=params)
    print(docs)
    
    # searcher = Agent(
    #     role="Web Research Expert",
    #     goal="Find related information from specific URL's",
    #     backstory="An expert web researcher that uses the web extremely well",
    #     tools=[spider_tool],
    #     verbose=True
    # )
    
    # summarize_spider = Task(
    #     description="Summarize the content of spider.cloud",
    #     expected_output="A summary that goes over what spider does",
    #     agent=searcher
    # )
    
    # crew = Crew(
    #     agents=[searcher],
    #     tasks=[summarize_spider],
    #     verbose=2
    # )
    
    # crew.kickoff()

if __name__ == "__main__":
    test_spider_tool()