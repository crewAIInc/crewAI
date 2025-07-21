"""
Simple test to verify the logging fix works without external API calls
"""
import logging
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter
from rich.tree import Tree


def test_console_formatter_logging_fix():
    """Test that ConsoleFormatter allows custom logging when Live session is active"""
    print("Testing ConsoleFormatter logging fix...")
    
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    log_buffer = io.StringIO()
    handler = logging.StreamHandler(log_buffer)
    formatter = logging.Formatter('CUSTOM_LOG: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    console_formatter = ConsoleFormatter(verbose=True)
    
    tree = Tree("Test Tree")
    console_formatter.print(tree)
    
    assert console_formatter._live is not None, "Live session should be active"
    
    logger.info("This should appear in the log buffer")
    
    log_output = log_buffer.getvalue()
    assert "CUSTOM_LOG: This should appear in the log buffer" in log_output, f"Custom log not found in output: {log_output}"
    
    assert console_formatter._live is not None, "Live session should still be active after custom logging"
    
    print("✅ ConsoleFormatter logging fix test passed!")
    
    logger.removeHandler(handler)
    handler.close()


def test_console_formatter_print_behavior():
    """Test that non-Tree content properly pauses/resumes Live sessions"""
    print("Testing ConsoleFormatter print behavior...")
    
    console_formatter = ConsoleFormatter(verbose=True)
    
    tree = Tree("Test Tree")
    console_formatter.print(tree)
    
    assert console_formatter._live is not None, "Live session should be active"
    
    stdout_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer):
        console_formatter.print("Non-tree content")
    
    output = stdout_buffer.getvalue()
    assert "Non-tree content" in output, f"Non-tree content not found in output: {output}"
    
    assert console_formatter._live is not None, "Live session should be restored after printing non-Tree content"
    
    print("✅ ConsoleFormatter print behavior test passed!")


if __name__ == "__main__":
    test_console_formatter_logging_fix()
    test_console_formatter_print_behavior()
    print("🎉 All simple logging fix tests passed!")
