"""Test for documentation links to ensure they are valid and accessible."""

import pytest
import requests
from urllib.parse import urlparse


class TestDocumentationLinks:
    """Test class for validating documentation links."""

    @pytest.mark.parametrize("url", [
        "https://github.com/AgentOps-AI/agentops/blob/main/examples/crewai/job_posting.py",
        "https://github.com/AgentOps-AI/agentops/blob/main/examples/crewai/markdown_validator.py",
    ])
    def test_agentops_example_links_are_accessible(self, url):
        """Test that AgentOps example links in documentation are accessible."""
        try:
            response = requests.get(url, timeout=10)
            assert response.status_code == 200, f"URL {url} returned status code {response.status_code}"
            
            parsed_url = urlparse(url)
            assert parsed_url.scheme in ['http', 'https'], f"URL {url} should use http or https protocol"
            assert parsed_url.netloc, f"URL {url} should have a valid domain"
            
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Failed to access URL {url}: {str(e)}")

    def test_agentops_examples_contain_agentops_implementation(self):
        """Test that the linked examples actually contain AgentOps implementation."""
        urls = [
            "https://raw.githubusercontent.com/AgentOps-AI/agentops/main/examples/crewai/job_posting.py",
            "https://raw.githubusercontent.com/AgentOps-AI/agentops/main/examples/crewai/markdown_validator.py",
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=10)
                assert response.status_code == 200, f"Could not fetch raw content from {url}"
                
                content = response.text.lower()
                assert "agentops" in content, f"Example at {url} does not contain AgentOps implementation"
                assert "import agentops" in content or "from agentops" in content, f"Example at {url} does not import AgentOps"
                
            except requests.exceptions.RequestException as e:
                pytest.fail(f"Failed to fetch content from {url}: {str(e)}")
