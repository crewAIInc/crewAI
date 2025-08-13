#!/usr/bin/env python3
"""Simple test to verify the verbose task name fix works."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from unittest.mock import Mock
from crewai.utilities.events.utils.console_formatter import ConsoleFormatter

def test_task_display_name():
    """Test the _get_task_display_name method directly."""
    print("Testing _get_task_display_name method...")
    
    formatter = ConsoleFormatter(verbose=True)
    
    task1 = Mock()
    task1.name = "Research Market Trends"
    task1.id = "12345678-1234-5678-9012-123456789abc"
    
    result1 = formatter._get_task_display_name(task1)
    print(f"Test 1 - Task with name: {result1}")
    assert "Research Market Trends" in result1
    assert "12345678" in result1
    print("✅ Test 1 passed")
    
    task2 = Mock()
    task2.name = None
    task2.description = "Analyze current market trends and provide insights"
    task2.id = "87654321-4321-8765-2109-987654321abc"
    
    result2 = formatter._get_task_display_name(task2)
    print(f"Test 2 - Task with description: {result2}")
    assert "Analyze current market trends" in result2
    assert "87654321" in result2
    print("✅ Test 2 passed")
    
    task3 = Mock()
    task3.name = None
    task3.description = None
    task3.id = "abcdef12-3456-7890-abcd-ef1234567890"
    
    result3 = formatter._get_task_display_name(task3)
    print(f"Test 3 - Task with ID only: {result3}")
    assert result3 == "abcdef12-3456-7890-abcd-ef1234567890"
    print("✅ Test 3 passed")
    
    print("\n🎉 All tests passed! The verbose task name fix is working correctly.")
    return True

if __name__ == "__main__":
    try:
        test_task_display_name()
        print("\n✅ Implementation verified successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
