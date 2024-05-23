from crewai_tools.tools.spider_full_tool.spider_full_tool import SpiderFullTool, SpiderFullParams

def test_spider_full_tool():
    spider_tool = SpiderFullTool(api_key="your_api_key")
    url = "https://spider.cloud"
    params = SpiderFullParams(
        request="http",
        limit=1,
        depth=1,
        cache=True,
        locale="en-US",
        stealth=True,
        headers={"User-Agent": "test-agent"},
        metadata=False,
        viewport="800x600",
        encoding="UTF-8",
        subdomains=False,
        user_agent="test-agent",
        store_data=False,
        proxy_enabled=False,
        query_selector=None,
        full_resources=False,
        request_timeout=30,
        run_in_background=False
    )
    docs = spider_tool._run(url=url, params=params)
    print(docs)

if __name__ == "__main__":
    test_spider_full_tool()
