from typing import Type, Optional, List, Dict, Any
import os
import json
from dotenv import load_dotenv

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import boto3
from botocore.exceptions import ClientError

# Load environment variables from .env file
load_dotenv()


class BedrockKBRetrieverToolInput(BaseModel):
    """Input schema for BedrockKBRetrieverTool."""
    query: str = Field(..., description="The query to retrieve information from the knowledge base")


class BedrockKBRetrieverTool(BaseTool):
    name: str = "Bedrock Knowledge Base Retriever Tool"
    description: str = "Retrieves information from an Amazon Bedrock Knowledge Base given a query"
    args_schema: Type[BaseModel] = BedrockKBRetrieverToolInput
    knowledge_base_id: str = None
    number_of_results: Optional[int] = 5
    retrieval_configuration: Optional[Dict[str, Any]] = None
    guardrail_configuration: Optional[Dict[str, Any]] = None
    next_token: Optional[str] = None

    def __init__(
        self,
        knowledge_base_id: str = None,
        number_of_results: Optional[int] = 5,
        retrieval_configuration: Optional[Dict[str, Any]] = None,
        guardrail_configuration: Optional[Dict[str, Any]] = None,
        next_token: Optional[str] = None,
        **kwargs
    ):
        """Initialize the BedrockKBRetrieverTool with knowledge base configuration.

        Args:
            knowledge_base_id (str): The unique identifier of the knowledge base to query (length: 0-10, pattern: ^[0-9a-zA-Z]+$)
            number_of_results (Optional[int], optional): The maximum number of results to return. Defaults to 5.
            retrieval_configuration (Optional[Dict[str, Any]], optional): Configurations for the knowledge base query and retrieval process. Defaults to None.
            guardrail_configuration (Optional[Dict[str, Any]], optional): Guardrail settings. Defaults to None.
            next_token (Optional[str], optional): Token for retrieving the next batch of results. Defaults to None.
        """
        super().__init__(**kwargs)
        
        # Get knowledge_base_id from environment variable if not provided
        self.knowledge_base_id = knowledge_base_id or os.getenv('BEDROCK_KB_ID')
        self.number_of_results = number_of_results
        
        # Initialize retrieval_configuration with number_of_results if provided
        if retrieval_configuration is None and number_of_results is not None:
            self.retrieval_configuration = {
                "vectorSearchConfiguration": {
                    "numberOfResults": number_of_results
                }
            }
        else:
            self.retrieval_configuration = retrieval_configuration
            
        self.guardrail_configuration = guardrail_configuration
        self.next_token = next_token

        # Validate parameters
        self._validate_parameters()

        # Update the description to include the knowledge base details
        self.description = f"Retrieves information from Amazon Bedrock Knowledge Base '{self.knowledge_base_id}' given a query"

    def _validate_parameters(self):
        """Validate the parameters according to AWS API requirements."""
        # Validate knowledge_base_id
        if not self.knowledge_base_id or len(self.knowledge_base_id) > 10 or not all(c.isalnum() for c in self.knowledge_base_id):
            raise ValueError("knowledge_base_id must be 0-10 alphanumeric characters")
        
        # Validate next_token if provided
        if self.next_token and (len(self.next_token) < 1 or len(self.next_token) > 2048 or ' ' in self.next_token):
            raise ValueError("next_token must be 1-2048 characters and match pattern ^\\S*$")

    def _run(self, query: str) -> str:
        try:
            # Initialize the Bedrock Agent Runtime client
            bedrock_agent_runtime = boto3.client(
                'bedrock-agent-runtime',
                region_name=os.getenv('AWS_REGION', os.getenv('AWS_DEFAULT_REGION', 'us-east-1')),
                # AWS SDK will automatically use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY from environment
            )

            # Prepare the request parameters
            retrieve_params = {
                'knowledgeBaseId': self.knowledge_base_id,
                'retrievalQuery': {
                    'text': query
                }
            }

            # Add optional parameters if provided
            if self.retrieval_configuration:
                retrieve_params['retrievalConfiguration'] = self.retrieval_configuration
                
            if self.guardrail_configuration:
                retrieve_params['guardrailConfiguration'] = self.guardrail_configuration
                
            if self.next_token:
                retrieve_params['nextToken'] = self.next_token

            # Make the retrieve API call
            response = bedrock_agent_runtime.retrieve(**retrieve_params)

            # Process the response
            results = []
            for result in response.get('retrievalResults', []):
                # Extract content
                content_obj = result.get('content', {})
                content = content_obj.get('text', '')
                content_type = content_obj.get('type', 'text')
                
                # Extract location information
                location = result.get('location', {})
                location_type = location.get('type', 'unknown')
                source_uri = None
                
                # Map for location types and their URI fields
                location_mapping = {
                    's3Location': {'field': 'uri', 'type': 'S3'},
                    'confluenceLocation': {'field': 'url', 'type': 'Confluence'},
                    'salesforceLocation': {'field': 'url', 'type': 'Salesforce'},
                    'sharePointLocation': {'field': 'url', 'type': 'SharePoint'},
                    'webLocation': {'field': 'url', 'type': 'Web'},
                    'customDocumentLocation': {'field': 'id', 'type': 'CustomDocument'},
                    'kendraDocumentLocation': {'field': 'uri', 'type': 'KendraDocument'},
                    'sqlLocation': {'field': 'query', 'type': 'SQL'}
                }
                
                # Extract the URI based on location type
                for loc_key, config in location_mapping.items():
                    if loc_key in location:
                        source_uri = location[loc_key].get(config['field'])
                        if not location_type or location_type == 'unknown':
                            location_type = config['type']
                        break
                
                # Include score if available
                score = result.get('score')
                
                # Include metadata if available
                metadata = result.get('metadata')
                
                # Create a well-formed JSON object for each result
                result_object = {
                    'content': content,
                    'content_type': content_type,
                    'source_type': location_type,
                    'source_uri': source_uri
                }
                
                # Add score if available
                if score is not None:
                    result_object['score'] = score
                
                # Add metadata if available
                if metadata:
                    result_object['metadata'] = metadata
                
                # Add the JSON object to results
                results.append(result_object)

            # Include nextToken in the response if available
            response_object = {}
            if results:
                response_object["results"] = results
            else:
                response_object["message"] = "No results found for the given query."
                
            if "nextToken" in response:
                response_object["nextToken"] = response["nextToken"]
                
            if "guardrailAction" in response:
                response_object["guardrailAction"] = response["guardrailAction"]

            # Return the results as a JSON string
            return json.dumps(response_object, indent=2)

        except ClientError as e:
            return f"Error retrieving from Bedrock Knowledge Base: {str(e)}"