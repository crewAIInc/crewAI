import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crewai.deployment.config import Config
from crewai.deployment.main import Deployment


class TestDeployment(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "config.yaml")
        
        # Create a test configuration file
        with open(self.config_path, "w") as f:
            f.write("""
            name: test-deployment
            port: 8000
            host: 127.0.0.1
            
            crews:
              - name: test_crew
                module_path: ./test_crew.py
                class_name: TestCrew
            """)
            
        # Create a test crew file
        with open(os.path.join(self.temp_dir.name, "test_crew.py"), "w") as f:
            f.write("""
            from crewai import Agent, Crew, Task
            
            class TestCrew:
                def crew(self):
                    return Crew(agents=[], tasks=[])
            """)
            
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_config_loading(self):
        config = Config(self.config_path)
        self.assertEqual(config.name, "test-deployment")
        self.assertEqual(config.port, 8000)
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(len(config.crews), 1)
        self.assertEqual(config.crews[0].name, "test_crew")
        
    @mock.patch("crewai.deployment.docker.container.DockerContainer.generate_dockerfile")
    @mock.patch("crewai.deployment.docker.container.DockerContainer.generate_compose_file")
    def test_deployment_prepare(self, mock_generate_compose, mock_generate_dockerfile):
        deployment = Deployment(self.config_path)
        deployment.deployment_dir = Path(self.temp_dir.name) / "deployment"
        
        deployment.prepare()
        
        # Check that the deployment directory was created
        self.assertTrue(os.path.exists(deployment.deployment_dir))
        
        # Check that the deployment config was created
        config_file = deployment.deployment_dir / "deployment_config.json"
        self.assertTrue(os.path.exists(config_file))
        
        # Check that Docker files were generated
        mock_generate_dockerfile.assert_called_once()
        mock_generate_compose.assert_called_once_with(port=8000)
