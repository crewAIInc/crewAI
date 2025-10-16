from crewai_tools import EXASearchTool

from crewai import LLM, Agent, Crew, Task
import os


llm = LLM(
    model="anthropic/claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)
agent = Agent(
    role="researcher",
    backstory="A researcher who can research the web",
    goal="Research the web",
    tools=[EXASearchTool()],
    llm=llm,
)

task = Task(
    description="Research the web based on the query: {query}",
    expected_output="A list of 10 bullet points of the most relevant information about the web",
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    verbose=True,
    tracing=True,
)

# result = crew.kickoff(inputs={"query": "What are ai agents?"})
# print("result", result)
# print("usage_metrics", result.token_usage)


def anthropic_tool_use_runner():
    def get_weather(location: str) -> str:
        return f"The weather in {location} is sunny"

    llm = LLM(
        model="anthropic/claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    result = llm.call(
        messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
        available_functions={"get_weather": get_weather},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The location to get the weather for",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ],
    )
    print("anthropic tool use result", result)


if __name__ == "__main__":
    anthropic_tool_use_runner()
