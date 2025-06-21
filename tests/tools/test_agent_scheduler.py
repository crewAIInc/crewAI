import os
import unittest
from crewai import Agent, Task, Crew, LLM
from crewAI.src.crewai.tools.agent_scheduler import AgentSchedulerTool
from langchain_openai import AzureChatOpenAI


os.environ["AZURE_API_TYPE"] = "azure"
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")
os.environ["AZURE_DEPLOYMENT_NAME"] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")


class TestAgentSchedulerTool(unittest.TestCase):
    def setup(self):
        self.tool = AgentSchedulerTool(agent_ids=["agent_alpha", "agent_beta", "agent_gamma"])
        self.llm = LLM(model="azure/gpt-4o", api_version="2023-05-15")

        self.agent = Agent(
            name="Scheduler Agent",
            role="Agent Performance Monitor",
            goal="Optimize agent retraining schedules based on recent outcomes",
            backstory="This agent reviews logs and adjusts how frequently agents should be retrained.",
            tools=[self.tool],
            llm=self.llm
        )

        self.task = Task(
            description="Use the agent_scheduler tool to analyze agent_alpha performance with 'True,False,True,True,False,False,True' and suggest a retraining interval.",
            expected_output="Suggest how often agent_alpha should be retrained",
            agent=self.agent
        )

    def test_tool_schema_structure(self):
        schema = self.tool.args_schema.schema()
        self.assertIn("agent_id", schema["properties"])
        self.assertIn("performance", schema["properties"])

    def test_agent_and_task_integration(self):
        self.assertEqual(self.agent.name, "Scheduler Agent")
        self.assertEqual(self.task.agent.name, "Scheduler Agent")
        self.assertTrue(any(isinstance(t, AgentSchedulerTool) for t in self.agent.tools))

    def test_crew_execution(self):
        crew = Crew(agents=[self.agent], tasks=[self.task], verbose=False)
        # This line will actually trigger execution. Comment if avoiding LLM calls.
        # print(agent.tools[0].args_schema.schema_json(indent=2))
        crew.kickoff()
        # self.assertIn("retrain", result.lower())
        # Instead, print schema for debug
        print(self.agent.tools[0].args_schema.schema_json(indent=2))


# ag = TestAgentSchedulerTool()
# ag.setup()
# ag.test_crew_execution()
