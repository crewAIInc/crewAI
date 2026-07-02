import pytest
import responses
from crewai_tools.tools.vault_search_tool.vault_search_tool import VaultSearchTool

class TestVaultSearchTool:
    """
    Unit tests for the VaultSearchTool class.
    
    These tests verify the tool's ability to handle successful API responses,
    empty results, server errors, and proper schema definition using 
    the 'responses' library to mock HTTP traffic.
    """

    @responses.activate
    def test_vault_search_successful_hit(self):
        """
        Verify the tool correctly parses and formats a successful search result.
        """
        # Register a mock rule for a successful POST request
        responses.add(
            responses.POST,
            "http://localhost:8000/vault/search",
            json={
                "hit": True,
                "results": [
                    {
                        "query_text": "How to configure VPC?",
                        "upvote_count": 15,
                        "downvote_count": 1,
                        "response_content": "Detailed VPC steps..."
                    }
                ]
            },
            status=200
        )

        # Initialize the tool and execute search
        tool = VaultSearchTool(api_url="http://localhost:8000")
        result = tool._run(query="VPC config")

        # Assertions to verify the formatted Markdown output
        assert "--- [Top Verified Report Found] ---" in result
        assert "Subject: How to configure VPC?" in result
        assert "👍15 Upvotes" in result
        assert "Detailed VPC steps..." in result

    @responses.activate
    def test_vault_search_no_results(self):
        """
        Verify the tool's behavior when the vault returns no matching reports.
        """
        responses.add(
            responses.POST,
            "http://localhost:8000/vault/search",
            json={"hit": False, "results": []},
            status=200
        )

        tool = VaultSearchTool(api_url="http://localhost:8000")
        result = tool._run(query="Unknown topic")

        assert result == "No verified reports found in the knowledge vault for this query."

    @responses.activate
    def test_vault_search_api_error(self):
        """
        Verify the tool gracefully handles HTTP error codes (e.g., 500 Internal Server Error).
        """
        responses.add(
            responses.POST,
            "http://localhost:8000/vault/search",
            status=500
        )

        tool = VaultSearchTool(api_url="http://localhost:8000")
        result = tool._run(query="Broken API")

        # The tool should return a descriptive error message instead of crashing
        assert "Error connecting to Knowledge Vault" in result
        assert "500" in result

    def test_tool_schema(self):
        """
        Verify that the tool inherits correctly from BaseTool and defines its schema properly.
        """
        tool = VaultSearchTool()
        assert tool.name == "Knowledge Vault Search"
        # Ensure 'query' is a required field in the Pydantic schema
        assert "query" in tool.args_schema.model_fields