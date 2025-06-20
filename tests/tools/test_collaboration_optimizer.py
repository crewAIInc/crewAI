import os
from crewai import Agent, Task, Crew, LLM
from crewAI.src.crewai.tools.collaboration_optimizer import CollaborationOptimizerTool
from langchain_openai import AzureChatOpenAI

# Set Azure OpenAI credentials
os.environ["AXZURE_API_TYPE"] = "azure"
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = "2025-01-01-preview"
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4o"


# llm = AzureChatOpenAI(azure_deployment='gpt-4o', azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'), api_key=os.getenv('AZURE_OPENAI_API_KEY'))
llm = LLM(model="azure/gpt-4o", api_version="2023-05-15")


class LLMWrapper:
    def __init__(self, llm): self.llm = llm
    def __call__(self, prompt): return self.llm.invoke(prompt).content


agent = Agent(
    name="Optimizer Agent",
    role="Collaboration Strategist",
    backstory="An AI agent specialized in optimizing teamwork among multiple agents through reinforcement learning strategies.",
    goal="Maximize team collaboration efficiency",
    tools=[CollaborationOptimizerTool()],
    llm=llm,
    verbose=True
)

task = Task(
    description="Run a simulation to optimize collaboration among 4 agents.",
    expected_output="Optimal reward score and strategy feedback",
    agent=agent
)

crew = Crew(agents=[agent], tasks=[task], verbose=True)
crew.kickoff()
