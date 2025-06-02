"""Unit tests for the ScratchpadTool."""

import pytest
from crewai.tools.agent_tools.scratchpad_tool import ScratchpadTool, ScratchpadToolSchema


class TestScratchpadTool:
    """Test suite for the ScratchpadTool functionality."""

    def test_schema_description(self):
        """Test that the schema has helpful description."""
        schema = ScratchpadToolSchema
        key_field = schema.model_fields['key']

        assert "Example:" in key_field.description
        assert '{"key":' in key_field.description

    def test_empty_scratchpad_error_message(self):
        """Test error message when scratchpad is empty."""
        tool = ScratchpadTool()
        result = tool._run(key="nonexistent")

        assert "❌ SCRATCHPAD IS EMPTY" in result
        assert "does not contain any data yet" in result
        assert "Try executing other tools first" in result
        assert "💡 TIP:" in result
        assert "search, read, or fetch operations" in result

    def test_key_not_found_error_message(self):
        """Test error message when key is not found."""
        tool = ScratchpadTool(scratchpad_data={
            "existing_key": "value",
            "another_key": {"data": "test"}
        })

        result = tool._run(key="wrong_key")

        assert "❌ KEY NOT FOUND: 'wrong_key'" in result
        assert "📦 AVAILABLE KEYS IN SCRATCHPAD:" in result
        assert "- 'existing_key'" in result
        assert "- 'another_key'" in result
        assert '✅ CORRECT USAGE EXAMPLE:' in result
        assert 'Action: Access Scratchpad Memory' in result
        assert 'Action Input: {"key": "existing_key"}' in result
        assert "⚠️ IMPORTANT:" in result
        assert "Keys are case-sensitive and must match EXACTLY" in result

    def test_successful_retrieval_string(self):
        """Test successful retrieval of string data."""
        tool = ScratchpadTool(scratchpad_data={
            "message": "Hello, World!"
        })

        result = tool._run(key="message")
        assert "✅ Successfully retrieved data for key 'message':" in result
        assert "Hello, World!" in result

    def test_successful_retrieval_dict(self):
        """Test successful retrieval of dictionary data."""
        test_dict = {"name": "John", "age": 30}
        tool = ScratchpadTool(scratchpad_data={
            "user_data": test_dict
        })

        result = tool._run(key="user_data")
        assert "✅ Successfully retrieved data for key 'user_data':" in result
        assert '"name": "John"' in result
        assert '"age": 30' in result

    def test_successful_retrieval_list(self):
        """Test successful retrieval of list data."""
        test_list = ["item1", "item2", "item3"]
        tool = ScratchpadTool(scratchpad_data={
            "items": test_list
        })

        result = tool._run(key="items")
        assert "✅ Successfully retrieved data for key 'items':" in result
        assert '"item1"' in result
        assert '"item2"' in result
        assert '"item3"' in result

    def test_tool_description_empty(self):
        """Test tool description when scratchpad is empty."""
        tool = ScratchpadTool()

        assert "HOW TO USE THIS TOOL:" in tool.description
        assert 'Example: {"key": "email_data"}' in tool.description
        assert "📝 STATUS: Scratchpad is currently empty" in tool.description

    def test_tool_description_with_data(self):
        """Test tool description when scratchpad has data."""
        tool = ScratchpadTool(scratchpad_data={
            "emails": ["email1@test.com", "email2@test.com"],
            "results": {"count": 5, "status": "success"},
            "api_key": "secret_key_123"
        })

        desc = tool.description

        # Check basic structure
        assert "HOW TO USE THIS TOOL:" in desc
        assert "📦 AVAILABLE DATA IN SCRATCHPAD:" in desc
        assert "💡 EXAMPLE USAGE:" in desc

        # Check key listings
        assert "📌 'emails': list of 2 items" in desc
        assert "📌 'results': dict with 2 items" in desc
        assert "📌 'api_key': string (14 chars)" in desc

        # Check example uses first key
        assert 'Action Input: {"key": "emails"}' in desc

    def test_update_scratchpad(self):
        """Test updating scratchpad data."""
        tool = ScratchpadTool()

        # Initially empty
        assert not tool.scratchpad_data

        # Update with data
        new_data = {"test": "value"}
        tool.update_scratchpad(new_data)

        assert tool.scratchpad_data == new_data
        assert "📌 'test': string (5 chars)" in tool.description

    def test_complex_data_preview(self):
        """Test preview generation for complex data structures."""
        tool = ScratchpadTool(scratchpad_data={
            "nested_dict": {
                "data": ["item1", "item2", "item3"]
            },
            "empty_list": [],
            "boolean_value": True,
            "number": 42
        })

        desc = tool.description

        # Special case for dict with 'data' key containing list
        assert "📌 'nested_dict': list of 3 items" in desc
        assert "📌 'empty_list': list of 0 items" in desc
        assert "📌 'boolean_value': bool" in desc
        assert "📌 'number': int" in desc

    def test_similar_key_suggestion(self):
        """Test that similar keys are suggested when a wrong key is used."""
        tool = ScratchpadTool(scratchpad_data={
            "email_search_results": ["email1", "email2"],
            "email_details": {"id": "123"},
            "user_preferences": {"theme": "dark"}
        })

        # Test partial match
        result = tool._run(key="email")
        assert "🔍 Did you mean one of these?" in result

        # Check that similar keys are in the suggestions
        # Extract just the "Did you mean" section
        did_you_mean_section = result.split("🔍 Did you mean one of these?")[1].split("✅ CORRECT USAGE EXAMPLE:")[0]
        assert "- 'email_search_results'" in did_you_mean_section
        assert "- 'email_details'" in did_you_mean_section
        assert "- 'user_preferences'" not in did_you_mean_section

        # But user_preferences should still be in the full list
        assert "- 'user_preferences'" in result

        # Test case-insensitive match
        result = tool._run(key="EMAIL_DETAILS")
        assert "🔍 Did you mean one of these?" in result
        assert "- 'email_details'" in result

        # Test no similar keys
        result = tool._run(key="completely_different")
        assert "🔍 Did you mean one of these?" not in result