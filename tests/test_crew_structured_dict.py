import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import json
import uuid

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.task import Task
from crewai.process import Process

class TestCrewStructuredDict(unittest.TestCase):
    def setUp(self):
        # Create test agents
        self.researcher = Agent(
            role="Researcher",
            goal="Research and gather information",
            backstory="You are an expert researcher with a keen eye for detail."
        )
        
        self.writer = Agent(
            role="Writer",
            goal="Write compelling content",
            backstory="You are a skilled writer who can create engaging content."
        )
        
        # Create test tasks
        self.research_task = Task(
            description="Research the latest AI developments",
            expected_output="A summary of the latest AI developments",
            agent=self.researcher
        )
        
        self.writing_task = Task(
            description="Write an article about AI developments",
            expected_output="A well-written article about AI developments",
            agent=self.writer,
            context=[self.research_task]
        )
        
        # Create a crew with tasks
        self.crew = Crew(
            agents=[self.researcher, self.writer],
            tasks=[self.research_task, self.writing_task],
            process=Process.sequential
        )
        
        # Create a crew with manager
        self.manager = Agent(
            role="Manager",
            goal="Manage the team",
            backstory="You are an experienced manager who oversees projects."
        )
        
        self.hierarchical_crew = Crew(
            agents=[self.researcher, self.writer],
            tasks=[self.research_task],
            process=Process.hierarchical,
            manager_agent=self.manager
        )
    
    def test_to_structured_dict_basic_properties(self):
        """Test that to_structured_dict includes basic Crew properties."""
        result = self.crew.to_structured_dict()
        
        # Check basic properties
        self.assertEqual(str(self.crew.id), result["id"])
        self.assertEqual(self.crew.name, result["name"])
        self.assertEqual(str(self.crew.process), result["process"])
        self.assertEqual(self.crew.verbose, result["verbose"])
        self.assertEqual(self.crew.memory, result["memory"])
    
    def test_to_structured_dict_agents(self):
        """Test that to_structured_dict includes agent information."""
        result = self.crew.to_structured_dict()
        
        # Check agents
        self.assertEqual(len(self.crew.agents), len(result["agents"]))
        
        # Check first agent properties
        agent_result = result["agents"][0]
        agent = self.crew.agents[0]
        self.assertEqual(str(agent.id), agent_result["id"])
        self.assertEqual(agent.role, agent_result["role"])
        self.assertEqual(agent.goal, agent_result["goal"])
        self.assertEqual(agent.backstory, agent_result["backstory"])
    
    def test_to_structured_dict_tasks(self):
        """Test that to_structured_dict includes task information."""
        result = self.crew.to_structured_dict()
        
        # Check tasks
        self.assertEqual(len(self.crew.tasks), len(result["tasks"]))
        
        # Check first task properties
        task_result = result["tasks"][0]
        task = self.crew.tasks[0]
        self.assertEqual(str(task.id), task_result["id"])
        self.assertEqual(task.description, task_result["description"])
        self.assertEqual(task.expected_output, task_result["expected_output"])
        self.assertEqual(task.agent.role, task_result["agent"])
    
    def test_to_structured_dict_task_relationships(self):
        """Test that to_structured_dict includes task relationships."""
        result = self.crew.to_structured_dict()
        
        # Check task relationships
        self.assertTrue("task_relationships" in result)
        self.assertEqual(1, len(result["task_relationships"]))
        
        # Check relationship properties
        relationship = result["task_relationships"][0]
        self.assertEqual(str(self.writing_task.id), relationship["task_id"])
        self.assertEqual(1, len(relationship["depends_on"]))
        self.assertEqual(str(self.research_task.id), relationship["depends_on"][0])
    
    def test_to_structured_dict_manager_agent(self):
        """Test that to_structured_dict includes manager agent information."""
        result = self.hierarchical_crew.to_structured_dict()
        
        # Check manager agent
        self.assertTrue("manager_agent" in result)
        manager_result = result["manager_agent"]
        self.assertEqual(str(self.manager.id), manager_result["id"])
        self.assertEqual(self.manager.role, manager_result["role"])
        self.assertEqual(self.manager.goal, manager_result["goal"])
        self.assertEqual(self.manager.backstory, manager_result["backstory"])
    
    def test_to_structured_dict_empty_tasks(self):
        """Test that to_structured_dict handles empty tasks list."""
        crew = Crew(
            agents=[self.researcher],
            tasks=[],
            process=Process.sequential
        )
        
        result = crew.to_structured_dict()
        
        # Check empty tasks
        self.assertEqual(0, len(result["tasks"]))
        self.assertEqual(0, len(result["task_relationships"]))
    
    def test_to_structured_dict_error_handling(self):
        """Test that to_structured_dict handles errors gracefully."""
        # Create a task with a valid agent
        task = Task(
            description="Test description",
            expected_output="Test output",
            agent=self.researcher
        )
        
        # Create a crew with the task
        crew = Crew(
            agents=[self.researcher],
            tasks=[task],
            process=Process.sequential
        )
        
        # Patch the task's context property to raise an exception
        # We'll do this by patching the specific method that accesses context
        with patch('crewai.crew.Crew._get_context', side_effect=Exception("Test exception")):
            # The method should not raise an exception
            result = crew.to_structured_dict()
            
            # Basic verification that the result still contains the task
            self.assertEqual(1, len(result["tasks"]))
            task_result = result["tasks"][0]
            self.assertEqual(str(task.id), task_result["id"])
            self.assertEqual(task.description, task_result["description"])
            self.assertEqual(task.expected_output, task_result["expected_output"])
            
            # Verify no task relationships were added due to the exception
            self.assertEqual(0, len(result["task_relationships"]))

if __name__ == "__main__":
    unittest.main()
