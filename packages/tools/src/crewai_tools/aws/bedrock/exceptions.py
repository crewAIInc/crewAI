"""Custom exceptions for AWS Bedrock integration."""

class BedrockError(Exception):
    """Base exception for Bedrock-related errors."""
    pass

class BedrockAgentError(BedrockError):
    """Exception raised for errors in the Bedrock Agent operations."""
    pass

class BedrockKnowledgeBaseError(BedrockError):
    """Exception raised for errors in the Bedrock Knowledge Base operations."""
    pass

class BedrockValidationError(BedrockError):
    """Exception raised for validation errors in Bedrock operations."""
    pass