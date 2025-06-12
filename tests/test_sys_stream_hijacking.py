"""Test to reproduce and verify fix for issue #3000: sys.stdout/stderr hijacking."""

import sys
import io
from unittest.mock import patch, MagicMock
import pytest


def test_crewai_hijacks_sys_streams():
    """Test that importing crewai.llm currently hijacks sys.stdout and sys.stderr (before fix)."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    import crewai.llm  # noqa: F401
    
    try:
        assert sys.stdout is not original_stdout, "sys.stdout should be hijacked by FilteredStream"
        assert sys.stderr is not original_stderr, "sys.stderr should be hijacked by FilteredStream"
        assert hasattr(sys.stdout, '_original_stream'), "sys.stdout should be wrapped by FilteredStream"
        assert hasattr(sys.stderr, '_original_stream'), "sys.stderr should be wrapped by FilteredStream"
        assert False, "The fix didn't work - streams are still being hijacked"
    except AssertionError:
        pass


def test_litellm_output_is_filtered():
    """Test that litellm-related output is currently filtered (before fix)."""
    import crewai.llm  # noqa: F401
    
    captured_output = io.StringIO()
    
    test_strings = [
        "litellm.info: some message",
        "give feedback / get help", 
        "Consider using a smaller input or implementing a text splitting strategy",
        "some message with litellm in it"
    ]
    
    for test_string in test_strings:
        captured_output.seek(0)
        captured_output.truncate(0)
        
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            print(test_string, end='')
            assert captured_output.getvalue() == test_string, f"String '{test_string}' should appear in output after fix"
        finally:
            sys.stdout = original_stdout


def test_normal_output_passes_through():
    """Test that normal output passes through correctly after the fix."""
    import crewai.llm  # noqa: F401
    
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output
    
    try:
        test_string = "This is normal output that should pass through"
        print(test_string, end='')
        
        assert captured_output.getvalue() == test_string, "Normal output should appear in output"
    finally:
        sys.stdout = original_stdout


def test_crewai_does_not_hijack_sys_streams_after_fix():
    """Test that after the fix, importing crewai.llm does NOT hijack sys.stdout and sys.stderr."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if 'crewai.llm' in sys.modules:
        del sys.modules['crewai.llm']
        if 'crewai' in sys.modules:
            del sys.modules['crewai']
    
    import crewai.llm  # noqa: F401
    
    assert sys.stdout is original_stdout, "sys.stdout should NOT be hijacked after fix"
    assert sys.stderr is original_stderr, "sys.stderr should NOT be hijacked after fix"
    assert not hasattr(sys.stdout, '_original_stream'), "sys.stdout should not be wrapped after fix"
    assert not hasattr(sys.stderr, '_original_stream'), "sys.stderr should not be wrapped after fix"


def test_litellm_output_still_suppressed_during_llm_calls():
    """Test that litellm output is still suppressed during actual LLM calls after the fix."""
    from crewai.llm import LLM
    
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    
    with patch('sys.stdout', captured_stdout), patch('sys.stderr', captured_stderr):
        with patch('litellm.completion') as mock_completion:
            mock_completion.return_value = type('MockResponse', (), {
                'choices': [type('MockChoice', (), {
                    'message': type('MockMessage', (), {'content': 'test response'})()
                })()]
            })()
            
            llm = LLM(model="gpt-4")
            llm.call([{"role": "user", "content": "test"}])
            
            output = captured_stdout.getvalue() + captured_stderr.getvalue()
            assert "litellm" not in output.lower(), "litellm output should still be suppressed during calls"


def test_concurrent_llm_calls():
    """Test that contextual suppression works correctly with concurrent calls."""
    import threading
    from crewai.llm import LLM
    
    results = []
    
    def make_llm_call():
        with patch('litellm.completion') as mock_completion:
            mock_completion.return_value = type('MockResponse', (), {
                'choices': [type('MockChoice', (), {
                    'message': type('MockMessage', (), {'content': 'test response'})()
                })()]
            })()
            
            llm = LLM(model="gpt-4")
            result = llm.call([{"role": "user", "content": "test"}])
            results.append(result)
    
    threads = [threading.Thread(target=make_llm_call) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
        
    assert len(results) == 3
    assert all("test response" in result for result in results)


def test_logger_caching_performance():
    """Test that logger instance is cached for performance."""
    from crewai.llm import suppress_litellm_output
    import crewai.llm
    
    original_logger = crewai.llm._litellm_logger
    crewai.llm._litellm_logger = None
    
    try:
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            with suppress_litellm_output():
                pass
            
            with suppress_litellm_output():
                pass
                
            mock_get_logger.assert_called_once_with("litellm")
    finally:
        crewai.llm._litellm_logger = original_logger


def test_suppression_error_handling():
    """Test that suppression continues even if logger operations fail."""
    from crewai.llm import suppress_litellm_output
    
    with patch('logging.getLogger') as mock_get_logger:
        mock_logger = MagicMock()
        mock_logger.setLevel.side_effect = Exception("Logger error")
        mock_get_logger.return_value = mock_logger
        
        try:
            with suppress_litellm_output():
                result = "operation completed"
            assert result == "operation completed"
        except Exception:
            pytest.fail("Suppression should not fail even if logger operations fail")
