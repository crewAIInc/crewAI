import logging
from datetime import datetime
from typing import Any

from botocore.client import Config
from pydantic import BaseModel, Field, field_validator

from crewai.memory.storage.interface import Storage

# Setup logger
logger = logging.getLogger(__name__)


class BedrockAgentCoreConfig(BaseModel):
    """Configuration for AgentCore memory storage."""

    # Required fields
    memory_id: str = Field(
        ..., description="AWS Bedrock AgentCore Memory ID (required)"
    )
    actor_id: str = Field(..., description="Actor ID for the memory session (required)")
    session_id: str = Field(
        ..., description="Session ID for the memory session (required)"
    )

    # Optional fields with defaults
    region_name: str = Field(
        default="us-east-1", description="AWS region name (default: 'us-east-1')"
    )

    # Namespace configuration
    namespaces: list[str] = Field(
        default_factory=list,
        description="List of namespaces to be searched (default: empty list)",
    )

    @field_validator("memory_id", "actor_id", "session_id")
    @classmethod
    def validate_required_ids(cls, v):
        if not v or not v.strip():
            raise ValueError("Required ID fields cannot be empty")
        return v.strip()

    @field_validator("region_name")
    @classmethod
    def validate_region(cls, v):
        if not v or not v.strip():
            raise ValueError("Region name cannot be empty")
        return v.strip()

    @field_validator("namespaces")
    @classmethod
    def validate_namespaces(cls, v):
        if not isinstance(v, list):
            raise ValueError("Namespaces must be a list")

        validated_namespaces = []
        for namespace in v:
            if not namespace or not namespace.strip():
                raise ValueError("Namespace cannot be empty")
            validated_namespaces.append(namespace.strip())

        return validated_namespaces

    class Config:
        """Pydantic configuration."""

        validate_assignment = True
        extra = "forbid"  # Prevent extra fields


class BedrockAgentCoreStorage(Storage):
    """
    AWS Bedrock AgentCore Memory storage implementation for CrewAI.

    This implementation provides integration with AWS Bedrock AgentCore Memory,
    supporting long term and short term memory. For more details, see the documentation:
    https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/memory.html.
    """

    def __init__(self, type: str, config: BedrockAgentCoreConfig):
        super().__init__()

        self._validate_type(type)
        self.memory_type = type
        self.config = config

        # Initialize boto3 clients
        self.memory_client: Any | None = None

        # Initialize memory clients
        self._initialize_memory_client()

        logger.info(
            "BedrockAgentCoreStorage initialized: memory_type=%s, namespaces=%s",
            self.memory_type,
            self.config.namespaces,
        )

    def _validate_type(self, type: str) -> None:
        """Validate the memory type."""
        supported_types = {"external"}
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for BedrockAgentCoreStorage. "
                f"Must be one of: {', '.join(supported_types)}"
            )

    def _initialize_memory_client(self) -> None:
        """Initialize the Bedrock AgentCore Memory client."""
        try:
            import boto3
        except ImportError as e:
            raise ModuleNotFoundError(
                "boto3 is required for Bedrock AgentCore. Install `crewai[agentcore]` for the required dependencies."
            ) from e

        try:
            # Initialize boto3 clients directly as used by the MemoryClient
            config = Config(user_agent_extra="x-client-framework:crew_ai")
            self.memory_client = boto3.client(
                "bedrock-agentcore",
                region_name=self.config.region_name,
                config=config,
            )

        except Exception as e:
            raise ValueError(
                f"Failed to initialize Bedrock AgentCore clients: {e!s}"
            ) from e

    def save(self, value: Any, metadata: dict[str, Any]) -> None:
        """
        Save memory item to AWS Bedrock AgentCore.

        Args:
            value: The content to save (task output, conversation, etc.) (required)
            metadata: Additional metadata for the memory item (default: None)
        """

        payload = self._convert_to_event_payload(value, metadata)

        try:
            # Use the boto3 client create_event method directly
            response = self.memory_client.create_event(  # type: ignore
                memoryId=self.config.memory_id,
                actorId=self.config.actor_id,
                sessionId=self.config.session_id,
                eventTimestamp=datetime.now(),
                payload=payload,
            )

            logger.info(
                "Created event in AgentCore memory: %s", response["event"]["eventId"]
            )

        except Exception as e:
            logger.error("Error saving to AgentCore: %s", str(e))
            raise

    def _search_long_term(
        self, query: str, limit: int, score_threshold: float
    ) -> list[dict[str, Any]]:
        """
        Search for memories in long-term storage (namespaces).

        Args:
            query: Search query
            limit: Maximum number of results
            score_threshold: Minimum relevance score

        Returns:
            List of memory items from long-term storage
        """
        namespaces_to_search = self.config.namespaces

        if not namespaces_to_search:
            logger.warning("No namespaces configured - no namespaces to search")
            return []

        logger.debug(
            "Searching configured namespaces: %s",
            namespaces_to_search,
        )

        long_term_memory_results = []

        # Search each namespace
        for search_namespace in namespaces_to_search:
            logger.debug("Searching namespace: %s", search_namespace)

            try:
                long_term_memory_response = self.memory_client.retrieve_memory_records(  # type: ignore
                    memoryId=self.config.memory_id,
                    namespace=search_namespace,
                    searchCriteria={"searchQuery": query, "topK": limit},
                )
            except Exception as e:
                logger.error(f"Error searching namespace {search_namespace}: {e}")
                raise

            long_term_memories = long_term_memory_response.get(
                "memoryRecordSummaries", []
            )

            logger.info(
                "Retrieved %d memories from namespace: %s",
                len(long_term_memories),
                search_namespace,
            )

            # Process search results
            for long_term_memory in long_term_memories:
                # Extract text content from memory record
                content = long_term_memory.get("content", {})
                text = content.get("text", "")

                # Get score if available
                score = long_term_memory.get("score", 0.0)

                # Apply score threshold filter
                if score < score_threshold:
                    continue

                long_term_memory_results.append(
                    {
                        "id": long_term_memory.get("memoryRecordId"),
                        "content": text,
                        "metadata": {
                            "namespaces": long_term_memory.get("namespaces"),
                            "created_at": long_term_memory.get("createdAt"),
                            "search_namespace": search_namespace,
                            "memory_strategy_id": long_term_memory.get(
                                "memoryStrategyId"
                            ),
                        },
                        "score": score,
                    }
                )

        # Sort by score and apply the overall limit
        long_term_memory_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # Apply the overall limit to the combined results
        long_term_memory_results = long_term_memory_results[:limit]

        logger.info(
            "AgentCore long-term search returned %d results from %d namespaces",
            len(long_term_memory_results),
            len(namespaces_to_search),
        )

        return long_term_memory_results

    def _search_short_term(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """
        Search for memories in short-term storage (events).

        Args:
            query: Search query (not used for short-term, but kept for consistency)
            max_results: Maximum number of results to retrieve

        Returns:
            List of memory items from short-term storage
        """
        short_term_results: list[dict[str, Any]] = []

        logger.info(
            "Searching short-term memory for up to %d results",
            max_results,
        )

        try:
            short_term_memory = self.memory_client.list_events(  # type: ignore
                memoryId=self.config.memory_id,
                actorId=self.config.actor_id,
                sessionId=self.config.session_id,
                includePayloads=True,
                maxResults=max_results,
            )
        except Exception as e:
            logger.error(f"Error listing events: {e}")
            raise

        for event in short_term_memory["events"]:
            for payload_item in event["payload"]:
                if "conversational" in payload_item:
                    text = payload_item["conversational"]["content"]["text"]
                    role = payload_item["conversational"]["role"]
                    short_term_results.append(
                        {
                            "id": event.get("eventId"),
                            "content": text,
                            "metadata": {
                                "created_at": event.get("eventTimestamp"),
                                "role": role,
                            },
                            "score": 1.0,  # Default score for short-term memory
                        }
                    )

        logger.info(
            "Retrieved %d results from short-term memory",
            len(short_term_results),
        )

        return short_term_results

    def search(
        self, query: str, limit: int = 5, score_threshold: float = 0.35
    ) -> list[dict[str, Any]]:
        """
        Search for memories using AgentCore boto3 client.

        Args:
            query: Search query (required)
            limit: Maximum number of results (default: 3, max: 100)
            score_threshold: Minimum relevance score (default: 0.35)

        Returns:
            List of memory items with context, sorted by relevance score
        """

        # Search long-term memory first
        long_term_results = self._search_long_term(query, limit, score_threshold)

        # Apply limit to long-term results
        final_results = long_term_results

        # If we have fewer results than requested
        # search short-term memory
        if len(final_results) < limit:
            remaining_limit = limit - len(final_results)
            logger.info(
                "Found %d long-term memory results, searching short-term memory for %d more",
                len(final_results),
                remaining_limit,
            )

            try:
                short_term_results = self._search_short_term(query, remaining_limit)
                final_results.extend(short_term_results)
            except Exception as e:
                logger.warning(f"Error searching short-term memory: {e}")
                # Continue with just long-term results

        logger.info(
            "Returning %d total results",
            len(final_results),
        )

        return final_results[:limit]

    def _safe_truncate_utf8(self, text: str, max_bytes: int = 9000) -> str:
        """
        Safely truncate a UTF-8 string to a maximum number of bytes without corrupting characters.
        Args:
            text: The text to truncate
            max_bytes: Maximum number of bytes (default: 9000)
        Returns:
            Truncated string that is valid UTF-8
        """
        if not text:
            return ""

        # Encode to UTF-8 bytes
        encoded = text.encode("utf-8")

        # If already within limit, return as-is
        if len(encoded) <= max_bytes:
            return text

        # Truncate bytes and decode, ignoring incomplete characters at the end
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")

        # Log if truncation occurred
        if len(truncated) < len(text):
            logger.debug(
                f"Truncated text from {len(text)} chars to {len(truncated)} chars"
            )

        return truncated

    def _convert_to_event_payload(
        self, value: Any, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Convert CrewAI content to event payload that utilizes metadata when available.

        Args:
            value: The content to convert
            metadata: metadata containing messages, agent, and description

        Returns:
            List of payload dictionaries for create_event
        """
        payload = []

        # Role mapping dictionary
        role_mapping = {
            "SYSTEM": "OTHER",  # AWS Bedrock AgentCore doesn't recognize SYSTEM
            "USER": "USER",
            "ASSISTANT": "ASSISTANT",
            "TOOL": "TOOL",
        }

        # Check if messages exist in metadata and process them
        messages = metadata.get("messages", [])

        if messages:
            # Process each message in the conversation
            for msg in messages:
                role = msg.get("role", "OTHER").upper()
                content = msg.get("content", "")

                payload_role = role_mapping.get(role, "OTHER")  # Default to OTHER

                # Safely truncate content to avoid UTF-8 corruption
                truncated_content = self._safe_truncate_utf8(content, 9000)

                payload.append(
                    {
                        "conversational": {
                            "content": {"text": truncated_content},
                            "role": payload_role,
                        }
                    }
                )

        # Always add the main value as an ASSISTANT message
        truncated_value = self._safe_truncate_utf8(str(value), 9000)
        payload.append(
            {
                "conversational": {
                    "content": {"text": truncated_value},
                    "role": "ASSISTANT",
                }
            }
        )

        logger.info(
            "Payload created: %d conversation messages + 1 task output = %d total items",
            len(messages),
            len(payload),
        )

        return payload

    def reset(self) -> None:
        """Reset/clear memory storage."""
        logger.warning(
            "Bedrock AgentCore memory reset not implemented - memories persist in AWS"
        )
        # Note: Bedrock AgentCore doesn't provide a direct reset API
        # delete_memory can be used to delete the entire resource
