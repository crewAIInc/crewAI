from typing import Type, Optional, Dict, Any
import os
import json
import uuid
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import boto3
from botocore.exceptions import ClientError

# Load environment variables from .env file
load_dotenv()


class BedrockInvokeAgentToolInput(BaseModel):
    """Input schema for BedrockInvokeAgentTool."""
    query: str = Field(..., description="The query to send to the agent")


class BedrockInvokeAgentTool(BaseTool):
    name: str = "Bedrock Agent Invoke Tool"
    description: str = "An agent responsible for policy analysis."
    args_schema: Type[BaseModel] = BedrockInvokeAgentToolInput
    agent_id: str = None
    agent_alias_id: str = None
    session_id: str = None
    enable_trace: bool = False
    end_session: bool = False

    def __init__(
        self,
        agent_id: str = None,
        agent_alias_id: str = None,
        session_id: str = None,
        enable_trace: bool = False,
        end_session: bool = False,
        description: Optional[str] = None,
        **kwargs
    ):
        """Initialize the BedrockInvokeAgentTool with agent configuration.

        Args:
            agent_id (str): The unique identifier of the Bedrock agent
            agent_alias_id (str): The unique identifier of the agent alias
            session_id (str): The unique identifier of the session
            enable_trace (bool): Whether to enable trace for the agent invocation
            end_session (bool): Whether to end the session with the agent
            description (Optional[str]): Custom description for the tool
        """
        super().__init__(**kwargs)
        
        # Get values from environment variables if not provided
        self.agent_id = agent_id or os.getenv('BEDROCK_AGENT_ID')
        self.agent_alias_id = agent_alias_id or os.getenv('BEDROCK_AGENT_ALIAS_ID')
        self.session_id = session_id or str(int(time.time()))  # Use timestamp as session ID if not provided
        self.enable_trace = enable_trace
        self.end_session = end_session

        # Update the description if provided
        if description:
            self.description = description

    def _run(self, query: str) -> str:
        try:
            # Initialize the Bedrock Agent Runtime client
            bedrock_agent = boto3.client(
                "bedrock-agent-runtime",
                region_name=os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-west-2'))
            )

            # Format the prompt with current time
            current_utc = datetime.now(timezone.utc)
            prompt = f"""
The current time is: {current_utc}

Below is the users query or task. Complete it and answer it consicely and to the point:
{query}
"""

            # Invoke the agent
            response = bedrock_agent.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=self.session_id,
                inputText=prompt,
                enableTrace=self.enable_trace,
                endSession=self.end_session
            )

            # Process the response
            completion = ""
            
            # Check if response contains a completion field
            if 'completion' in response:
                # Process streaming response format
                for event in response.get('completion', []):
                    if 'chunk' in event and 'bytes' in event['chunk']:
                        chunk_bytes = event['chunk']['bytes']
                        if isinstance(chunk_bytes, (bytes, bytearray)):
                            completion += chunk_bytes.decode('utf-8')
                        else:
                            completion += str(chunk_bytes)
            
            # If no completion found in streaming format, try direct format
            if not completion and 'chunk' in response and 'bytes' in response['chunk']:
                chunk_bytes = response['chunk']['bytes']
                if isinstance(chunk_bytes, (bytes, bytearray)):
                    completion = chunk_bytes.decode('utf-8')
                else:
                    completion = str(chunk_bytes)
            
            # If still no completion, return debug info
            if not completion:
                debug_info = {
                    "error": "Could not extract completion from response",
                    "response_keys": list(response.keys())
                }
                
                # Add more debug info
                if 'chunk' in response:
                    debug_info["chunk_keys"] = list(response['chunk'].keys())
                
                return json.dumps(debug_info, indent=2)
            
            return completion

        except ClientError as e:
            error_code = "Unknown"
            error_message = str(e)
            
            # Try to extract error code if available
            if hasattr(e, 'response') and 'Error' in e.response and 'Code' in e.response['Error']:
                error_code = e.response['Error']['Code']
            
            return f"Error invoking Bedrock Agent ({error_code}): {error_message}"
        except Exception as e:
            return f"Error: {str(e)}"