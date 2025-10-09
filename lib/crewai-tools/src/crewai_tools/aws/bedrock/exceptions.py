"""Custom exceptions for AWS Bedrock integration."""


class BedrockError(Exception):
    """Base exception for Bedrock-related errors."""


class BedrockAgentError(BedrockError):
    """Exception raised for errors in the Bedrock Agent operations."""


class BedrockKnowledgeBaseError(BedrockError):
    """Exception raised for errors in the Bedrock Knowledge Base operations."""


class BedrockValidationError(BedrockError):
    """Exception raised for validation errors in Bedrock operations."""
