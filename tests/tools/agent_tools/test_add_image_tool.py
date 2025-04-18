import os
import base64
import pytest
from unittest.mock import patch, MagicMock

from crewai.tools.agent_tools.add_image_tool import AddImageTool


class TestAddImageTool:
    def setup_method(self):
        self.tool = AddImageTool()
        os.makedirs("tests/tools/agent_tools/test_files", exist_ok=True)
    
    def test_add_image_with_url(self):
        result = self.tool._run(image_url="https://example.com/image.jpg")
        assert result["role"] == "user"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "image_url"
        assert result["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"
    
    def test_add_image_with_local_file(self):
        test_file_path = "tests/tools/agent_tools/test_files/test_image.jpg"
        
        with patch("builtins.open", MagicMock()), \
             patch("base64.b64encode", return_value=b"test_encoded_content"), \
             patch("os.path.exists", return_value=True):
            
            result = self.tool._run(image_url=test_file_path)
            
            assert result["role"] == "user"
            assert len(result["content"]) == 2
            assert result["content"][0]["type"] == "text"
            assert result["content"][1]["type"] == "image_url"
            assert result["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    
    def test_add_image_with_claude_3_7_model(self):
        mock_llm = MagicMock()
        mock_llm.model = "claude-3-7-sonnet-latest"
        
        with patch("os.path.exists", return_value=False):
            result = self.tool._run(
                image_url="https://example.com/image.jpg",
                llm=mock_llm
            )
            
            assert result["role"] == "user"
            assert len(result["content"]) == 2
            assert result["content"][0]["type"] == "text"
            assert result["content"][1]["type"] == "image_url"
            assert result["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"
    
    def test_add_image_with_invalid_path(self):
        with pytest.raises(ValueError):
            with patch("os.path.exists", return_value=True), \
                 patch("builtins.open", side_effect=FileNotFoundError()):
                self.tool._run(image_url="/invalid/path/to/image.jpg")
