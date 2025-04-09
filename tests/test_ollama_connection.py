import pytest
from unittest.mock import patch, MagicMock
from crewai.llm import LLM
from crewai.utilities.exceptions.ollama_connection_exception import OllamaConnectionException

class TestOllamaConnection:
    def test_ollama_connection_error(self):
        with patch('litellm.completion') as mock_completion:
            mock_completion.side_effect = Exception("OllamaException - [Errno 111] Connection refused")
            
            llm = LLM(model="ollama/llama3")
            
            with pytest.raises(OllamaConnectionException) as exc_info:
                llm.call([{"role": "user", "content": "Hello"}])
            
            assert "Failed to connect to Ollama" in str(exc_info.value)
            assert "Please make sure Ollama is installed and running" in str(exc_info.value)
