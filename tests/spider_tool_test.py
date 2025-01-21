from crewai import Agent, Crew, Task

from crewai_tools.tools.spider_tool.spider_tool import SpiderTool


def test_spider_tool():
    spider_tool = SpiderTool()

    searcher = Agent(
        role="Web Research Expert",
        goal="Find related information from specific URL's",
        backstory="An expert web researcher that uses the web extremely well",
        tools=[spider_tool],
        verbose=True,
        cache=False,
    )

    choose_between_scrape_crawl = Task(
        description="Scrape the page of spider.cloud and return a summary of how fast it is",
        expected_output="spider.cloud is a fast scraping and crawling tool",
        agent=searcher,
    )

    return_metadata = Task(
        description="Scrape https://spider.cloud with a limit of 1 and enable metadata",
        expected_output="Metadata and 10 word summary of spider.cloud",
        agent=searcher,
    )

    css_selector = Task(
        description="Scrape one page of spider.cloud with the `body > div > main > section.grid.md\:grid-cols-2.gap-10.place-items-center.md\:max-w-screen-xl.mx-auto.pb-8.pt-20 > div:nth-child(1) > h1` CSS selector",
        expected_output="The content of the element with the css selector body > div > main > section.grid.md\:grid-cols-2.gap-10.place-items-center.md\:max-w-screen-xl.mx-auto.pb-8.pt-20 > div:nth-child(1) > h1",
        agent=searcher,
    )

    crew = Crew(
        agents=[searcher],
        tasks=[choose_between_scrape_crawl, return_metadata, css_selector],
        verbose=True,
    )

    crew.kickoff()


if __name__ == "__main__":
    test_spider_tool()
