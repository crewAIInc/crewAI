"""AWS Bedrock System Tools Configuration.

This module provides configuration helpers for AWS Bedrock system tools,
which are built-in tools provided by AWS for specific functionality like web grounding.

System tools are executed by AWS Bedrock internally, not by CrewAI.
They are configuration objects passed to the Bedrock Converse API.

When system tools are used, the response structure changes:
- Without system tools: Returns plain string (text only)
- With system tools: Returns the complete raw Bedrock response dict:
  {
    "ResponseMetadata": {...},      # AWS response metadata
    "output": {
      "message": {
        "role": "assistant",
        "content": [...]              # Array of content blocks (text, citationsContent, toolUse, toolResult, etc.)
      }
    },
    "stopReason": "end_turn",        # Why generation stopped
    "usage": {...},                  # Token usage statistics
    "metrics": {...},                # Performance metrics
    "processed_text": "..."          # Convenience field: combined text after hooks
  }

The response preserves the exact structure returned by AWS Bedrock, allowing users
to extract any fields they need without transformation.
"""

from typing import Any, Dict


def create_nova_web_grounding_config() -> Dict[str, Any]:
    """Create configuration for Nova Web Grounding system tool.
    
    Nova Web Grounding enables Amazon Nova models to search the web for current
    information and provide responses with citations. This feature is useful for
    queries requiring up-to-date information beyond the model's training data.
    
    Requirements:
        - Only available in US regions with US CRIS profiles
        - Only works with Nova models (e.g., us.amazon.nova-2-lite-v1:0)
        - Requires bedrock:InvokeTool permission for the nova_grounding system tool
    
    Returns:
        Dictionary with systemTool configuration for Bedrock Converse API
        
    Example:
        ```python
        from crewai import LLM
        from crewai.llms.providers.bedrock.system_tools import create_nova_web_grounding_config
        
        # Create LLM with system tools
        llm = LLM(
            model="bedrock/us.amazon.nova-2-lite-v1:0",
            region_name="us-east-1",
            system_tools=[create_nova_web_grounding_config()]
        )
        
        # Call returns full raw Bedrock response
        response = llm.call("What are the latest AI developments?")
        
        # Access the raw Bedrock structure
        output = response["output"]
        message = output["message"]
        content_blocks = message["content"]
        
        # Or use the convenience field
        text = response["processed_text"]
        
        # Extract citations from content blocks
        for block in content_blocks:
            if "citationsContent" in block:
                citations = block["citationsContent"]["citations"]
                for citation in citations:
                    location = citation.get("location", {})
                    web = location.get("web", {})
                    print(f"URL: {web.get('url')}")
                    print(f"Domain: {web.get('domain')}")
            
            if "text" in block:
                print(f"Text: {block['text']}")
        
        # Access usage statistics
        usage = response["usage"]
        print(f"Tokens used: {usage['totalTokens']}")
        ```
    
    References:
        - https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
        - https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
    """
    return {
        "systemTool": {
            "name": "nova_grounding"
        }
    }


__all__ = [
    "create_nova_web_grounding_config",
]
