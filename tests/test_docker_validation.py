"""Test Docker validation functionality in Agent."""

import os
import subprocess
from unittest.mock import Mock, patch, mock_open
import pytest
from crewai import Agent


class TestDockerValidation:
    """Test cases for Docker validation in Agent."""

    def test_docker_validation_skipped_with_env_var(self):
        """Test that Docker validation is skipped when CREWAI_SKIP_DOCKER_VALIDATION=true."""
        with patch.dict(os.environ, {"CREWAI_SKIP_DOCKER_VALIDATION": "true"}):
            agent = Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory",
                allow_code_execution=True,
            )
            assert agent.allow_code_execution is True

    def test_docker_validation_skipped_with_unsafe_mode(self):
        """Test that Docker validation is skipped when code_execution_mode='unsafe'."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal", 
            backstory="Test backstory",
            allow_code_execution=True,
            code_execution_mode="unsafe",
        )
        assert agent.code_execution_mode == "unsafe"

    @patch("crewai.agent.os.path.exists")
    def test_docker_validation_skipped_in_container_dockerenv(self, mock_exists):
        """Test that Docker validation is skipped when /.dockerenv exists."""
        mock_exists.return_value = True
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory", 
            allow_code_execution=True,
        )
        assert agent.allow_code_execution is True

    @patch("crewai.agent.os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="12:memory:/docker/container123")
    def test_docker_validation_skipped_in_container_cgroup(self, mock_file, mock_exists):
        """Test that Docker validation is skipped when cgroup indicates container."""
        mock_exists.return_value = False
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            allow_code_execution=True,
        )
        assert agent.allow_code_execution is True

    @patch("crewai.agent.os.path.exists")
    @patch("crewai.agent.os.getpid")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_docker_validation_skipped_in_container_pid1(self, mock_file, mock_getpid, mock_exists):
        """Test that Docker validation is skipped when running as PID 1."""
        mock_exists.return_value = False
        mock_getpid.return_value = 1
        
        agent = Agent(
            role="Test Agent", 
            goal="Test goal",
            backstory="Test backstory",
            allow_code_execution=True,
        )
        assert agent.allow_code_execution is True

    @patch("crewai.agent.shutil.which")
    @patch("crewai.agent.os.path.exists")
    @patch("crewai.agent.os.getpid")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_docker_validation_fails_no_docker(self, mock_file, mock_getpid, mock_exists, mock_which):
        """Test that Docker validation fails when Docker is not installed."""
        mock_exists.return_value = False
        mock_getpid.return_value = 1000
        mock_which.return_value = None
        
        with pytest.raises(RuntimeError, match="Docker is not installed"):
            Agent(
                role="Test Agent",
                goal="Test goal", 
                backstory="Test backstory",
                allow_code_execution=True,
            )

    @patch("crewai.agent.shutil.which")
    @patch("crewai.agent.subprocess.run")
    @patch("crewai.agent.os.path.exists")
    @patch("crewai.agent.os.getpid")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_docker_validation_fails_docker_not_running(self, mock_file, mock_getpid, mock_exists, mock_run, mock_which):
        """Test that Docker validation fails when Docker daemon is not running."""
        mock_exists.return_value = False
        mock_getpid.return_value = 1000
        mock_which.return_value = "/usr/bin/docker"
        mock_run.side_effect = subprocess.CalledProcessError(1, "docker info")
        
        with pytest.raises(RuntimeError, match="Docker is not running"):
            Agent(
                role="Test Agent",
                goal="Test goal",
                backstory="Test backstory", 
                allow_code_execution=True,
            )

    @patch("crewai.agent.shutil.which")
    @patch("crewai.agent.subprocess.run")
    @patch("crewai.agent.os.path.exists")
    @patch("crewai.agent.os.getpid")
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_docker_validation_passes_docker_available(self, mock_file, mock_getpid, mock_exists, mock_run, mock_which):
        """Test that Docker validation passes when Docker is available."""
        mock_exists.return_value = False
        mock_getpid.return_value = 1000
        mock_which.return_value = "/usr/bin/docker"
        mock_run.return_value = Mock(returncode=0)
        
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            allow_code_execution=True,
        )
        assert agent.allow_code_execution is True

    def test_container_detection_methods(self):
        """Test the container detection logic directly."""
        agent = Agent(
            role="Test Agent",
            goal="Test goal", 
            backstory="Test backstory",
        )
        
        with patch("crewai.agent.os.path.exists", return_value=True):
            assert agent._is_running_in_container() is True
            
        with patch("crewai.agent.os.path.exists", return_value=False), \
             patch("builtins.open", mock_open(read_data="docker")):
            assert agent._is_running_in_container() is True
            
        with patch("crewai.agent.os.path.exists", return_value=False), \
             patch("builtins.open", side_effect=FileNotFoundError), \
             patch("crewai.agent.os.getpid", return_value=1):
            assert agent._is_running_in_container() is True

    def test_reproduce_original_issue(self):
        """Test that reproduces the original issue from GitHub issue #3028."""
        with patch("crewai.agent.os.path.exists", return_value=True):
            agent = Agent(
                role="Knowledge Pattern Synthesizer",
                goal="Synthesize knowledge patterns",
                backstory="You're an expert at synthesizing knowledge patterns.",
                allow_code_execution=True,
                verbose=True,
                memory=True,
                max_retry_limit=3
            )
            assert agent.allow_code_execution is True
            assert agent.role == "Knowledge Pattern Synthesizer"
