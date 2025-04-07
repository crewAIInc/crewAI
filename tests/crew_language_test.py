import pytest
from unittest.mock import patch

from crewai import Crew, Agent, Process, Task
from crewai.utilities.i18n import I18N


def test_crew_with_language():
    i18n = I18N(language="en")
    
    agent = Agent(
        role="Test Agent",
        goal="Test Goal",
        backstory="Test Backstory",
        verbose=True
    )
    
    task = Task(
        description="Test Task",
        expected_output="Test Output",
        agent=agent
    )
    
    with patch('crewai.crew.I18N') as mock_i18n:
        mock_i18n.return_value = i18n
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            language="fr"  # Use French as an example
        )
        
        with patch.object(crew, '_run_sequential_process'):
            with patch.object(crew, '_set_tasks_callbacks'):
                with patch('crewai.agent.Agent.create_agent_executor'):
                    crew.kickoff()
                    
                    mock_i18n.assert_called_with(prompt_file=None, language="fr")
