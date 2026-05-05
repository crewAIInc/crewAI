from typing import Any, Optional, Type
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class VaultSearchSchema(BaseModel):
    """Input schema for VaultSearchTool."""
    query: str = Field(..., description="The semantic search query to look up in the knowledge vault.")

class VaultSearchTool(BaseTool):
    """
    A search tool for retrieving team-verified technical reports from a Knowledge Vault.
    
    This tool performs semantic search against a centralized repository of peer-validated 
    content, ensuring that agents prioritize historical team consensus over general 
    internet search results.
    """
    name: str = "Knowledge Vault Search"
    description: str = (
        "Useful for retrieving verified technical reports and team consensus. "
        "Use this to find historical context and peer-validated research before "
        "conducting new external searches."
    )
    args_schema: Type[BaseModel] = VaultSearchSchema
    
    api_url: str = Field(
        default="http://localhost:8000", 
        description="The base URL of the Knowledge Vault API server."
    )

    def _run(self, query: str) -> str:
        """
        Execute a POST request to the vault server and format the top result.

        Args:
            query (str): The search string provided by the agent.

        Returns:
            str: A formatted string containing the top search hit, community scores, 
                 and a content summary, or an error/empty message.
        """
        try:
            # Send search request to the vault server
            # The server expects a JSON payload with 'query', 'threshold', and 'match_count'
            response = requests.post(
                f"{self.api_url.rstrip('/')}/vault/search",
                json={
                    "query": query, 
                    "threshold": 0.7, 
                    "match_count": 5
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract results based on the standard server response format: {"hit": bool, "results": list}
            results = data.get("results", [])
            
            if not results:
                return "No verified reports found in the knowledge vault for this query."

            top = results[0]
            
            # Format the output for the CrewAI Agent to ingest
            return (
                f"--- [Top Verified Report Found] ---\n"
                f"Subject: {top.get('query_text', 'N/A')}\n"
                f"Community Score: 👍{top.get('upvote_count', 0)} Upvotes | 👎{top.get('downvote_count', 0)} Downvotes\n"
                f"Content Summary:\n{top.get('response_content', '')[:1500]}\n"
                f"--- End of Report ---"
            )

        except requests.exceptions.RequestException as e:
            return f"Error connecting to Knowledge Vault: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred while searching the vault: {str(e)}"