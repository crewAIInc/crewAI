from crewai import LLM
from crewai.lite_agent import LiteAgent
from crewai.tools import BaseTool


# A simple test tool
class SecretLookupTool(BaseTool):
    name = "secret_lookup"
    description = "A tool to lookup secrets"

    def _run(self) -> str:
        return "SUPERSECRETPASSWORD123"


# Test with tools
def test_with_tools():
    llm = LLM(model="gpt-4o")
    agent = LiteAgent(
        role="Secret Agent",
        goal="Return the secret password",
        backstory="I am a secret agent created to return the secret password",
        llm=llm,
        tools=[SecretLookupTool()],
        verbose=True,
    )

    # Test a simple query
    response = agent.kickoff("Hello, can you help me?")
    print("\n=== Agent Response ===")
    print(response)


# # Test without tools
# def test_without_tools():
#     llm = LLM(model="gpt-4o")
#     agent = LiteAgent(
#         role="Test Agent",
#         goal="Test the system prompt formatting",
#         backstory="I am a test agent created to verify the system prompt works correctly.",
#         llm=llm,
#         verbose=True,
#     )

#     # Get the system prompt
#     system_prompt = agent._get_default_system_prompt()
#     print("\n=== System Prompt (without tools) ===")
#     print(system_prompt)

#     # Test a simple query
#     response = agent.kickoff("Hello, can you help me?")
#     print("\n=== Agent Response ===")
#     print(response)


if __name__ == "__main__":
    print("Testing LiteAgent with tools...")
    test_with_tools()

    # print("\n\nTesting LiteAgent without tools...")
    # test_without_tools()
