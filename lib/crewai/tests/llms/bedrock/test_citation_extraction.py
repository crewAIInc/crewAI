"""Tests for citation extraction from Nova Web Grounding responses."""

import pytest
from unittest.mock import Mock, patch
from crewai.llms.providers.bedrock.completion import BedrockCompletion


class TestCitationExtraction:
    """Test citation extraction from Bedrock responses."""

    @pytest.fixture
    def bedrock_llm(self):
        """Create a BedrockCompletion instance for testing."""
        with patch("crewai.llms.providers.bedrock.completion.Session"):
            llm = BedrockCompletion(
                model="us.amazon.nova-2-lite-v1:0",
                region_name="us-east-1",
            )
            return llm

    def test_extract_citations_from_response(self, bedrock_llm):
        """Test that citations are extracted from response content blocks."""
        # Mock response with citations
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": "Recent quantum computing developments include",
                            "citationsContent": {
                                "citations": [
                                    {
                                        "location": {
                                            "web": {
                                                "url": "https://example.com/quantum",
                                                "domain": "example.com"
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 20,
                "totalTokens": 30
            }
        }

        # Mock the client.converse call
        bedrock_llm.client.converse = Mock(return_value=mock_response)

        # Call the LLM
        result = bedrock_llm.call(
            messages="What are recent quantum computing developments?",
            tools=[{"systemTool": {"name": "nova_grounding"}}]
        )

        # Verify citation is appended to text
        assert "https://example.com/quantum" in result
        assert "[https://example.com/quantum]" in result

    def test_extract_multiple_citations(self, bedrock_llm):
        """Test extraction of multiple citations from a single response."""
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": "Information from multiple sources",
                            "citationsContent": {
                                "citations": [
                                    {
                                        "location": {
                                            "web": {
                                                "url": "https://source1.com",
                                                "domain": "source1.com"
                                            }
                                        }
                                    },
                                    {
                                        "location": {
                                            "web": {
                                                "url": "https://source2.com",
                                                "domain": "source2.com"
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}
        }

        bedrock_llm.client.converse = Mock(return_value=mock_response)

        result = bedrock_llm.call(
            messages="Test query",
            tools=[{"systemTool": {"name": "nova_grounding"}}]
        )

        # Verify both citations are present
        assert "[https://source1.com]" in result
        assert "[https://source2.com]" in result

    def test_response_without_citations(self, bedrock_llm):
        """Test that responses without citations work normally."""
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": "Response without citations"
                        }
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}
        }

        bedrock_llm.client.converse = Mock(return_value=mock_response)

        result = bedrock_llm.call(
            messages="Test query",
            tools=[{"systemTool": {"name": "nova_grounding"}}]
        )

        # Verify text is returned without errors
        assert result == "Response without citations"
        assert "[" not in result  # No citation markers

    def test_empty_citations_list(self, bedrock_llm):
        """Test handling of empty citations list."""
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": "Response text",
                            "citationsContent": {
                                "citations": []
                            }
                        }
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}
        }

        bedrock_llm.client.converse = Mock(return_value=mock_response)

        result = bedrock_llm.call(
            messages="Test query",
            tools=[{"systemTool": {"name": "nova_grounding"}}]
        )

        assert result == "Response text"

    def test_malformed_citation_structure(self, bedrock_llm):
        """Test handling of malformed citation structures."""
        mock_response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": "Response text",
                            "citationsContent": {
                                "citations": [
                                    {
                                        "location": {
                                            # Missing 'web' key
                                            "other": {}
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 20, "totalTokens": 30}
        }

        bedrock_llm.client.converse = Mock(return_value=mock_response)

        # Should not raise an error, just skip malformed citation
        result = bedrock_llm.call(
            messages="Test query",
            tools=[{"systemTool": {"name": "nova_grounding"}}]
        )

        assert result == "Response text"
        assert "[" not in result
