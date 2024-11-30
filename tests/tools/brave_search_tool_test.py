from crewai_tools.tools.brave_search_tool.brave_search_tool import BraveSearchTool


def test_brave_tool():
    tool = BraveSearchTool(
        n_results=2,
    )

    print(tool.run(search_query="ChatGPT"))


if __name__ == "__main__":
    test_brave_tool()
