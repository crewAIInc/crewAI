import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
from click.testing import CliRunner

from crewai.cli.cli import create
from crewai.cli.create_crew import create_crew


class TestCreateCrew(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    @patch("crewai.cli.create_crew.get_provider_data")
    @patch("crewai.cli.create_crew.select_provider")
    @patch("crewai.cli.create_crew.select_model")
    @patch("crewai.cli.create_crew.write_env_file")
    @patch("crewai.cli.create_crew.load_env_vars")
    @patch("click.confirm")
    def test_create_crew_handles_unicode(self, mock_confirm, mock_load_env, 
                                         mock_write_env, mock_select_model, 
                                         mock_select_provider, mock_get_provider_data):
        """Test that create_crew command handles Unicode properly."""
        mock_confirm.return_value = True
        mock_load_env.return_value = {}
        mock_get_provider_data.return_value = {"openai": ["gpt-4"]}
        mock_select_provider.return_value = "openai"
        mock_select_model.return_value = "gpt-4"
        
        templates_dir = Path("src/crewai/cli/templates/crew")
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        template_content = """
        Hello {{name}}! Unicode test: 你好, こんにちは, Привет 🚀
        Class: {{crew_name}}
        Folder: {{folder_name}}
        """
        
        (templates_dir / "tools").mkdir(exist_ok=True)
        (templates_dir / "config").mkdir(exist_ok=True)
        
        for file_name in [".gitignore", "pyproject.toml", "README.md", "__init__.py", "main.py", "crew.py"]:
            with open(templates_dir / file_name, "w", encoding="utf-8") as f:
                f.write(template_content)
                
        (templates_dir / "knowledge").mkdir(exist_ok=True)
        with open(templates_dir / "knowledge" / "user_preference.txt", "w", encoding="utf-8") as f:
            f.write(template_content)
            
        for file_path in ["tools/custom_tool.py", "tools/__init__.py", "config/agents.yaml", "config/tasks.yaml"]:
            (templates_dir / file_path).parent.mkdir(exist_ok=True, parents=True)
            with open(templates_dir / file_path, "w", encoding="utf-8") as f:
                f.write(template_content)
        
        with patch("crewai.cli.create_crew.Path") as mock_path:
            mock_path.return_value = self.test_dir
            mock_path.side_effect = lambda x: self.test_dir / x if isinstance(x, str) else x
            
            create_crew("test_crew", skip_provider=True)
            
        crew_dir = self.test_dir / "test_crew"
        for root, _, files in os.walk(crew_dir):
            for file in files:
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    self.assertIn("你好", content, f"Unicode characters not preserved in {file_path}")
                    self.assertIn("🚀", content, f"Emoji not preserved in {file_path}")
