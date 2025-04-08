from typing import Type, Optional, List, Dict, Any
import os
import json
from dotenv import load_dotenv

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ..exceptions import BedrockKnowledgeBaseError, BedrockValidationError

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
            knowledge_base_id (str): The unique identifier of the knowledge base to query
            number_of_results (Optional[int], optional): The maximum number of results to return. Defaults to 5.
            retrieval_configuration (Optional[Dict[str, Any]], optional): Configurations for the knowledge base query and retrieval process. Defaults to None.
            guardrail_configuration (Optional[Dict[str, Any]], optional): Guardrail settings. Defaults to None.
            next_token (Optional[str], optional): Token for retrieving the next batch of results. Defaults to None.
        """
        super().__init__(**kwargs)
        
        # Get knowledge_base_id from environment variable if not provided
        self.knowledge_base_id = knowledge_base_id or os.getenv('BEDROCK_KB_ID')
        self.number_of_results = number_of_results
        self.guardrail_configuration = guardrail_configuration
        self.next_token = next_token
        
        # Initialize retrieval_configuration with provided parameters or use the one provided
        if retrieval_configuration is None:
            self.retrieval_configuration = self._build_retrieval_configuration()
        else:
            self.retrieval_configuration = retrieval_configuration

        # Validate parameters
        self._validate_parameters()

        # Update the description to include the knowledge base details
        self.description = f"Retrieves information from Amazon Bedrock Knowledge Base '{self.knowledge_base_id}' given a query"

    def _build_retrieval_configuration(self) -> Dict[str, Any]:
        """Build the retrieval configuration based on provided parameters.
        
        Returns:
            Dict[str, Any]: The constructed retrieval configuration
        """
        vector_search_config = {}
        
        # Add number of results if provided
        if self.number_of_results is not None:
            vector_search_config["numberOfResults"] = self.number_of_results
            
        return {"vectorSearchConfiguration": vector_search_config}

    def _validate_parameters(self):
        """Validate the parameters according to AWS API requirements."""
        try:
            # Validate knowledge_base_id
            if not self.knowledge_base_id:
                raise BedrockValidationError("knowledge_base_id cannot be empty")
            if not isinstance(self.knowledge_base_id, str):
                raise BedrockValidationError("knowledge_base_id must be a string")
            if len(self.knowledge_base_id) > 10:
                raise BedrockValidationError("knowledge_base_id must be 10 characters or less")
            if not all(c.isalnum() for c in self.knowledge_base_id):
                raise BedrockValidationError("knowledge_base_id must contain only alphanumeric characters")
            
            # Validate next_token if provided
            if self.next_token:
                if not isinstance(self.next_token, str):
                    raise BedrockValidationError("next_token must be a string")
                if len(self.next_token) < 1 or len(self.next_token) > 2048:
                    raise BedrockValidationError("next_token must be between 1 and 2048 characters")
                if ' ' in self.next_token:
                    raise BedrockValidationError("next_token cannot contain spaces")
                    
            # Validate number_of_results if provided
            if self.number_of_results is not None:
                if not isinstance(self.number_of_results, int):
                    raise BedrockValidationError("number_of_results must be an integer")
                if self.number_of_results < 1:
                    raise BedrockValidationError("number_of_results must be greater than 0")
                    
        except BedrockValidationError as e:
            raise BedrockValidationError(f"Parameter validation failed: {str(e)}")

    def _process_retrieval_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single retrieval result from Bedrock Knowledge Base.
        
        Args:
            result (Dict[str, Any]): Raw result from Bedrock Knowledge Base
            
        Returns:
            Dict[str, Any]: Processed result with standardized format
        """
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
        
        # Create result object
        result_object = {
            'content': content,
            'content_type': content_type,
            'source_type': location_type,
            'source_uri': source_uri
        }
        
        # Add optional fields if available
        if 'score' in result:
            result_object['score'] = result['score']
        
        if 'metadata' in result:
            result_object['metadata'] = result['metadata']
            
        # Handle byte content if present
        if 'byteContent' in content_obj:
            result_object['byte_content'] = content_obj['byteContent']
            
        # Handle row content if present
        if 'row' in content_obj:
            result_object['row_content'] = content_obj['row']
            
        return result_object

    def _run(self, query: str) -> str:
        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            raise ImportError("`boto3` package not found, please run `uv add boto3`")

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
                processed_result = self._process_retrieval_result(result)
                results.append(processed_result)

            # Build the response object
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
            error_code = "Unknown"
            error_message = str(e)
            
            # Try to extract error code if available
            if hasattr(e, 'response') and 'Error' in e.response:
                error_code = e.response['Error'].get('Code', 'Unknown')
                error_message = e.response['Error'].get('Message', str(e))
            
            raise BedrockKnowledgeBaseError(f"Error ({error_code}): {error_message}")
        except Exception as e:
            raise BedrockKnowledgeBaseError(f"Unexpected error: {str(e)}")