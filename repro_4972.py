import unittest
from typing import Any
from crewai.agents.crew_agent_executor import CrewAgentExecutor

class TestParseNativeToolCall(unittest.TestCase):
    def setUp(self):
        # We need a mock agent or just call the method on a dummy instance
        self.executor = CrewAgentExecutor.__new__(CrewAgentExecutor)

    def test_parse_bedrock_converse_format(self):
        # Format reported in issue #4972
        tool_call = {
            "toolUse": {
                "toolUseId": "tooluse_123",
                "name": "calculator",
                "input": {"expression": "1 + 1"}
            }
        }
        
        result = self.executor._parse_native_tool_call(tool_call)
        
        self.assertIsNotNone(result, "Should not return None for Bedrock format")
        call_id, func_name, func_args = result
        self.assertEqual(call_id, "tooluse_123")
        self.assertEqual(func_name, "calculator")
        self.assertEqual(func_args, {"expression": "1 + 1"})

    def test_parse_direct_bedrock_format(self):
        # Format if already unwrapped
        tool_call = {
            "toolUseId": "tooluse_123",
            "name": "calculator",
            "input": {"expression": "1 + 1"}
        }
        
        result = self.executor._parse_native_tool_call(tool_call)
        
        self.assertIsNotNone(result)
        call_id, func_name, func_args = result
        self.assertEqual(call_id, "tooluse_123")
        self.assertEqual(func_name, "calculator")
        self.assertEqual(func_args, {"expression": "1 + 1"})

if __name__ == "__main__":
    unittest.main()
