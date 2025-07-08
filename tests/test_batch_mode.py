import pytest
from unittest.mock import Mock, patch
from crewai.llm import LLM


class TestBatchMode:
    """Test suite for Google Batch Mode functionality."""

    def test_batch_mode_initialization(self):
        """Test that batch mode parameters are properly initialized."""
        llm = LLM(
            model="gemini/gemini-1.5-pro",
            batch_mode=True,
            batch_size=5,
            batch_timeout=600
        )
        
        assert llm.batch_mode is True
        assert llm.batch_size == 5
        assert llm.batch_timeout == 600
        assert llm._batch_requests == []
        assert llm._current_batch_job is None

    def test_batch_mode_defaults(self):
        """Test default values for batch mode parameters."""
        llm = LLM(model="gemini/gemini-1.5-pro", batch_mode=True)
        
        assert llm.batch_mode is True
        assert llm.batch_size == 10
        assert llm.batch_timeout == 300

    def test_is_gemini_model_detection(self):
        """Test Gemini model detection for batch mode support."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            llm_gemini = LLM(model="gemini/gemini-1.5-pro")
            assert llm_gemini._is_gemini_model() is True
            
            llm_openai = LLM(model="gpt-4")
            assert llm_openai._is_gemini_model() is False

    def test_is_gemini_model_without_genai_available(self):
        """Test Gemini model detection when google-generativeai is not available."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', False):
            llm = LLM(model="gemini/gemini-1.5-pro")
            assert llm._is_gemini_model() is False

    def test_prepare_batch_request(self):
        """Test batch request preparation."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            llm = LLM(
                model="gemini/gemini-1.5-pro",
                temperature=0.7,
                top_p=0.9,
                max_tokens=1000
            )
            
            messages = [{"role": "user", "content": "Hello, world!"}]
            batch_request = llm._prepare_batch_request(messages)
            
            assert "model" in batch_request
            assert batch_request["model"] == "gemini-1.5-pro"
            assert "contents" in batch_request
            assert "generationConfig" in batch_request
            assert batch_request["generationConfig"]["temperature"] == 0.7
            assert batch_request["generationConfig"]["topP"] == 0.9
            assert batch_request["generationConfig"]["maxOutputTokens"] == 1000

    def test_prepare_batch_request_non_gemini_model(self):
        """Test that batch request preparation fails for non-Gemini models."""
        llm = LLM(model="gpt-4")
        messages = [{"role": "user", "content": "Hello, world!"}]
        
        with pytest.raises(ValueError, match="Batch mode is only supported for Gemini models"):
            llm._prepare_batch_request(messages)

    @patch('crewai.llm.genai')
    def test_submit_batch_job(self, mock_genai):
        """Test batch job submission."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            mock_batch_job = Mock()
            mock_batch_job.name = "test-job-123"
            mock_genai.create_batch_job.return_value = mock_batch_job
            
            llm = LLM(
                model="gemini/gemini-1.5-pro",
                api_key="test-key"
            )
            
            requests = [{"model": "gemini-1.5-pro", "contents": []}]
            job_name = llm._submit_batch_job(requests)
            
            assert job_name == "test-job-123"
            mock_genai.configure.assert_called_with(api_key="test-key")
            mock_genai.create_batch_job.assert_called_once()

    def test_submit_batch_job_without_genai(self):
        """Test batch job submission without google-generativeai available."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', False):
            llm = LLM(model="gemini/gemini-1.5-pro")
            
            with pytest.raises(ImportError, match="google-generativeai is required for batch mode"):
                llm._submit_batch_job([])

    def test_submit_batch_job_without_api_key(self):
        """Test batch job submission without API key."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            llm = LLM(model="gemini/gemini-1.5-pro")
            
            with pytest.raises(ValueError, match="API key is required for batch mode"):
                llm._submit_batch_job([])

    @patch('crewai.llm.genai')
    @patch('crewai.llm.time')
    def test_poll_batch_job_success(self, mock_time, mock_genai):
        """Test successful batch job polling."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            mock_batch_job = Mock()
            mock_batch_job.state = "JOB_STATE_SUCCEEDED"
            mock_genai.get_batch_job.return_value = mock_batch_job
            mock_time.time.side_effect = [0, 1, 2]
            mock_time.sleep = Mock()
            
            llm = LLM(
                model="gemini/gemini-1.5-pro",
                api_key="test-key"
            )
            
            result = llm._poll_batch_job("test-job-123")
            
            assert result == mock_batch_job
            mock_genai.get_batch_job.assert_called_with("test-job-123")

    @patch('crewai.llm.genai')
    @patch('crewai.llm.time')
    def test_poll_batch_job_timeout(self, mock_time, mock_genai):
        """Test batch job polling timeout."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            mock_batch_job = Mock()
            mock_batch_job.state = "JOB_STATE_PENDING"
            mock_genai.get_batch_job.return_value = mock_batch_job
            mock_time.time.side_effect = [0, 400]
            mock_time.sleep = Mock()
            
            llm = LLM(
                model="gemini/gemini-1.5-pro",
                api_key="test-key",
                batch_timeout=300
            )
            
            with pytest.raises(TimeoutError, match="did not complete within 300 seconds"):
                llm._poll_batch_job("test-job-123")

    @patch('crewai.llm.genai')
    def test_retrieve_batch_results(self, mock_genai):
        """Test batch result retrieval."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            mock_batch_job = Mock()
            mock_batch_job.state = "JOB_STATE_SUCCEEDED"
            mock_genai.get_batch_job.return_value = mock_batch_job
            
            mock_response = Mock()
            mock_response.response.candidates = [Mock()]
            mock_response.response.candidates[0].content.parts = [Mock()]
            mock_response.response.candidates[0].content.parts[0].text = "Test response"
            
            mock_genai.list_batch_job_responses.return_value = [mock_response]
            
            llm = LLM(
                model="gemini/gemini-1.5-pro",
                api_key="test-key"
            )
            
            results = llm._retrieve_batch_results("test-job-123")
            
            assert results == ["Test response"]
            mock_genai.get_batch_job.assert_called_with("test-job-123")
            mock_genai.list_batch_job_responses.assert_called_with("test-job-123")

    @patch('crewai.llm.genai')
    def test_retrieve_batch_results_failed_job(self, mock_genai):
        """Test batch result retrieval for failed job."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            mock_batch_job = Mock()
            mock_batch_job.state = "JOB_STATE_FAILED"
            mock_genai.get_batch_job.return_value = mock_batch_job
            
            llm = LLM(
                model="gemini/gemini-1.5-pro",
                api_key="test-key"
            )
            
            with pytest.raises(RuntimeError, match="Batch job failed with state: JOB_STATE_FAILED"):
                llm._retrieve_batch_results("test-job-123")

    @patch('crewai.llm.crewai_event_bus')
    def test_handle_batch_request_non_gemini(self, mock_event_bus):
        """Test batch request handling for non-Gemini models."""
        llm = LLM(model="gpt-4", batch_mode=True)
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="Batch mode is only supported for Gemini models"):
            llm._handle_batch_request(messages)

    @patch('crewai.llm.crewai_event_bus')
    def test_batch_mode_call_routing(self, mock_event_bus):
        """Test that batch mode calls are routed correctly."""
        with patch('crewai.llm.GOOGLE_GENAI_AVAILABLE', True):
            llm = LLM(
                model="gemini/gemini-1.5-pro",
                batch_mode=True,
                api_key="test-key"
            )
            
            with patch.object(llm, '_handle_batch_request') as mock_batch_handler:
                mock_batch_handler.return_value = "Batch response"
                
                result = llm.call("Hello, world!")
                
                assert result == "Batch response"
                mock_batch_handler.assert_called_once()

    def test_non_batch_mode_unchanged(self):
        """Test that non-batch mode behavior is unchanged."""
        with patch('crewai.llm.litellm') as mock_litellm:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Regular response"
            mock_response.choices[0].message.tool_calls = []
            mock_litellm.completion.return_value = mock_response
            
            llm = LLM(model="gemini/gemini-1.5-pro", batch_mode=False)
            result = llm.call("Hello, world!")
            
            assert result == "Regular response"
            mock_litellm.completion.assert_called_once()
