"""Tests for MLFlow integration."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys


class TestMLFlowIntegration:
    """Test MLFlow integration functionality."""

    def test_is_mlflow_available_when_installed(self):
        """Test that is_mlflow_available returns True when MLFlow is installed."""
        with patch.dict(sys.modules, {'mlflow': Mock()}):
            from crewai.integrations.mlflow import is_mlflow_available
            assert is_mlflow_available() is True

    def test_is_mlflow_available_when_not_installed(self):
        """Test that is_mlflow_available returns False when MLFlow is not installed."""
        with patch.dict(sys.modules, {'mlflow': None}):
            with patch('crewai.integrations.mlflow.logger'):
                from crewai.integrations.mlflow import is_mlflow_available
                assert is_mlflow_available() is False

    def test_setup_mlflow_autolog_success(self):
        """Test successful MLFlow autolog setup."""
        mock_mlflow = Mock()
        mock_mlflow.crewai.autolog = Mock()
        
        with patch.dict(sys.modules, {'mlflow': mock_mlflow}):
            from crewai.integrations.mlflow import setup_mlflow_autolog
            result = setup_mlflow_autolog()
            
            assert result is True
            mock_mlflow.crewai.autolog.assert_called_once_with(
                log_traces=True,
                log_models=False,
                disable=False,
                exclusive=False,
                disable_for_unsupported_versions=False,
                silent=False,
            )

    def test_setup_mlflow_autolog_not_available(self):
        """Test MLFlow autolog setup when MLFlow is not available."""
        with patch.dict(sys.modules, {'mlflow': None}):
            with patch('crewai.integrations.mlflow.logger') as mock_logger:
                from crewai.integrations.mlflow import setup_mlflow_autolog
                result = setup_mlflow_autolog()
                
                assert result is False
                mock_logger.warning.assert_called_once()

    def test_setup_mlflow_autolog_silent_mode(self):
        """Test MLFlow autolog setup in silent mode."""
        with patch.dict(sys.modules, {'mlflow': None}):
            with patch('crewai.integrations.mlflow.logger') as mock_logger:
                from crewai.integrations.mlflow import setup_mlflow_autolog
                result = setup_mlflow_autolog(silent=True)
                
                assert result is False
                mock_logger.warning.assert_not_called()

    def test_setup_mlflow_autolog_exception(self):
        """Test MLFlow autolog setup when an exception occurs."""
        mock_mlflow = Mock()
        mock_mlflow.crewai.autolog.side_effect = Exception("Test error")
        
        with patch.dict(sys.modules, {'mlflow': mock_mlflow}):
            with patch('crewai.integrations.mlflow.logger') as mock_logger:
                from crewai.integrations.mlflow import setup_mlflow_autolog
                result = setup_mlflow_autolog()
                
                assert result is False
                mock_logger.error.assert_called_once()

    def test_get_active_run_success(self):
        """Test getting active MLFlow run."""
        mock_run = Mock()
        mock_mlflow = Mock()
        mock_mlflow.active_run.return_value = mock_run
        
        with patch.dict(sys.modules, {'mlflow': mock_mlflow}):
            from crewai.integrations.mlflow import get_active_run
            result = get_active_run()
            
            assert result == mock_run
            mock_mlflow.active_run.assert_called_once()

    def test_get_active_run_not_available(self):
        """Test getting active MLFlow run when MLFlow is not available."""
        with patch.dict(sys.modules, {'mlflow': None}):
            from crewai.integrations.mlflow import get_active_run
            result = get_active_run()
            
            assert result is None

    def test_get_active_run_exception(self):
        """Test getting active MLFlow run when an exception occurs."""
        mock_mlflow = Mock()
        mock_mlflow.active_run.side_effect = Exception("Test error")
        
        with patch.dict(sys.modules, {'mlflow': mock_mlflow}):
            from crewai.integrations.mlflow import get_active_run
            result = get_active_run()
            
            assert result is None

    def test_log_crew_execution_success(self):
        """Test logging crew execution to MLFlow."""
        mock_mlflow = Mock()
        mock_context_manager = Mock()
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_context_manager)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        
        with patch.dict(sys.modules, {'mlflow': mock_mlflow}):
            from crewai.integrations.mlflow import log_crew_execution
            log_crew_execution("test_crew", param1="value1", param2="value2")
            
            mock_mlflow.start_run.assert_called_once_with(run_name="crew_test_crew")
            assert mock_mlflow.log_param.call_count == 2

    def test_log_crew_execution_not_available(self):
        """Test logging crew execution when MLFlow is not available."""
        with patch.dict(sys.modules, {'mlflow': None}):
            from crewai.integrations.mlflow import log_crew_execution
            log_crew_execution("test_crew", param1="value1")

    def test_log_crew_execution_exception(self):
        """Test logging crew execution when an exception occurs."""
        mock_mlflow = Mock()
        mock_mlflow.start_run.side_effect = Exception("Test error")
        
        with patch.dict(sys.modules, {'mlflow': mock_mlflow}):
            with patch('crewai.integrations.mlflow.logger') as mock_logger:
                from crewai.integrations.mlflow import log_crew_execution
                log_crew_execution("test_crew", param1="value1")
                mock_logger.debug.assert_called_once()


class TestMLFlowAutologIntegration:
    """Test the actual MLFlow autolog integration."""

    @pytest.mark.skipif(
        not pytest.importorskip("mlflow", minversion="2.19.0"),
        reason="MLFlow not available or version too old"
    )
    def test_mlflow_crewai_autolog_exists(self):
        """Test that mlflow.crewai.autolog exists and can be called."""
        try:
            import mlflow
            assert hasattr(mlflow, 'crewai')
            assert hasattr(mlflow.crewai, 'autolog')
            
            mlflow.crewai.autolog(disable=True)  # Disable to avoid side effects
            
        except ImportError:
            pytest.skip("MLFlow not available")
        except Exception as e:
            pytest.fail(f"mlflow.crewai.autolog() failed: {e}")

    def test_mlflow_integration_with_crew(self):
        """Test MLFlow integration with CrewAI Crew class."""
        mock_mlflow = Mock()
        mock_mlflow.crewai.autolog = Mock()
        
        with patch.dict(sys.modules, {'mlflow': mock_mlflow}):
            from crewai.integrations.mlflow import setup_mlflow_autolog
            from crewai import Crew, Agent, Task
            
            setup_mlflow_autolog()
            
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
            assert len(crew.agents) == 1
            assert len(crew.tasks) == 1

    def test_documentation_example(self):
        """Test the example from the documentation."""
        mock_mlflow = Mock()
        mock_mlflow.crewai.autolog = Mock()
        mock_mlflow.set_tracking_uri = Mock()
        
        with patch.dict(sys.modules, {'mlflow': mock_mlflow}):
            import mlflow
            
            mlflow.set_tracking_uri("http://localhost:5000")
            
            mlflow.crewai.autolog()
            
            mock_mlflow.set_tracking_uri.assert_called_once_with("http://localhost:5000")
            mock_mlflow.crewai.autolog.assert_called_once()
