import os
import unittest
from crewai import Agent, Task, Crew, LLM
from crewAI.src.crewai.tools.collaboration_optimizer import CollaborationOptimizerTool

# Set Azure OpenAI credentials
os.environ["AZURE_API_TYPE"] = "azure"
os.environ["AZURE_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ["AZURE_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["AZURE_API_VERSION"] = "2025-01-01-preview"
os.environ["AZURE_DEPLOYMENT_NAME"] = "gpt-4o"


class TestCollaborationOptimizerTool(unittest.TestCase):
    def setup(self):
        self.llm = LLM(model="azure/gpt-4o", api_version="2023-05-15")
        self.agent = Agent(
            name="Optimizer Agent",
            role="Collaboration Strategist",
            backstory="An AI agent specialized in optimizing teamwork among multiple agents through reinforcement learning strategies.",
            goal="Maximize team collaboration efficiency",
            tools=[CollaborationOptimizerTool()],
            llm=self.llm,
            verbose=True
        )

        self.task = Task(
            description="Run a simulation to optimize collaboration among 4 agents.",
            expected_output="Optimal reward score and strategy feedback",
            agent=self.agent
        )

        self.crew = Crew(agents=[self.agent], tasks=[self.task], verbose=True)

      def test_collaboration_optimizer_tool_attached_to_agent(self):
        # Ensure the tool is properly attached
        tool_names = [tool.name for tool in self.agent.tools]
        self.assertIn("Collaboration Optimizer", tool_names)

      def test_crew_kickoff_returns_result(self):
          # Run the crew and assert the result format
          result = self.crew.kickoff()
          self.assertIsInstance(result, str)  # Or dict, depending on what the tool returns
          self.assertIn("Optimal", result)  # You can refine this based on expected output

      def test_tool_description_contains_expected_keywords(self):
          tool = self.agent.tools[0]
          self.assertIn("optimiz", tool.description.lower())  # fuzzy match for "optimize", etc.
          self.assertTrue(tool.description)  # Ensure description is not empty

