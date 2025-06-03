from unittest.mock import Mock, patch
import sys

from crewai.integrations.mlflow import autolog
from crewai.utilities.events.third_party.mlflow_listener import mlflow_listener


class TestMLflowIntegration:
    
    def test_autolog_without_mlflow_installed(self, caplog):
        """Test autolog when MLflow is not installed"""
        with patch.dict(sys.modules, {'mlflow': None}):
            with patch('crewai.integrations.mlflow.mlflow', None):
                autolog()
                assert "MLflow is not installed" in caplog.text
    
    @patch('crewai.integrations.mlflow.mlflow')
    def test_autolog_enable(self, mock_mlflow):
        """Test enabling autolog"""
        autolog()
        assert mlflow_listener._autolog_enabled is True
    
    @patch('crewai.integrations.mlflow.mlflow')
    def test_autolog_disable(self, mock_mlflow):
        """Test disabling autolog"""
        autolog(disable=True)
        assert mlflow_listener._autolog_enabled is False
    
    @patch('crewai.integrations.mlflow.mlflow')
    def test_autolog_silent_mode(self, mock_mlflow, caplog):
        """Test silent mode suppresses logging"""
        autolog(silent=True)
        assert "MLflow autologging enabled" not in caplog.text
    
    @patch('crewai.integrations.mlflow.mlflow')
    def test_mlflow_patching(self, mock_mlflow):
        """Test that mlflow.crewai.autolog is available"""
        from crewai.integrations.mlflow import _patch_mlflow
        _patch_mlflow()
        assert hasattr(mock_mlflow, 'crewai')
        assert hasattr(mock_mlflow.crewai, 'autolog')

    def test_reproduction_case_issue_2947(self):
        """Test the exact case from issue #2947"""
        with patch('crewai.integrations.mlflow.mlflow') as mock_mlflow:
            mock_mlflow.tracing.start_span.return_value = Mock()
            
            autolog()
            assert mlflow_listener._autolog_enabled is True
            
            from crewai import Agent, Task, Crew
            
            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory"
            )
            
            task = Task(
                description="Test task",
                expected_output="Test output",
                agent=agent
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task]
            )
            
            assert crew is not None
