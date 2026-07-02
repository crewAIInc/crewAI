"""Live demonstration of SearchApiSearchTool with multiple engines.

Run with: SEARCHAPI_API_KEY=your_key uv run python examples/searchapi_tool_demo.py
"""

import json
import os
import sys

sys.path.insert(0, "lib/crewai-tools/src")
sys.path.insert(0, "lib/crewai/src")

from crewai_tools import SearchApiSearchTool


def demo_engine(engine: str, query: str) -> None:
    print(f"\n{'='*60}")
    print(f"Engine: {engine}")
    print(f"Query:  {query}")
    print("=" * 60)

    tool = SearchApiSearchTool(engine=engine, n_results=3)
    result = tool.run(search_query=query)

    if isinstance(result, str):
        print(f"Error: {result}")
        return

    print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])
    print(f"\n... ({len(json.dumps(result))} total chars)")


def main() -> None:
    if not os.getenv("SEARCHAPI_API_KEY"):
        print("Set SEARCHAPI_API_KEY environment variable to run this demo.")
        print("Get a free key at https://www.searchapi.io")
        sys.exit(1)

    demos = [
        ("google", "CrewAI AI agents framework"),
        ("google_news", "artificial intelligence 2026"),
        ("google_shopping", "mechanical keyboard"),
        ("youtube", "python tutorial beginners"),
    ]

    for engine, query in demos:
        demo_engine(engine, query)


if __name__ == "__main__":
    main()
