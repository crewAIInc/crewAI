import logging
from typing import Any, Dict, List

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
    namespaces: List[str] = Field(
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
    Enhanced AWS Bedrock AgentCore Memory storage implementation for CrewAI.

    This implementation provides sophisticated integration with AWS Bedrock AgentCore Memory,
    supporting advanced features like user preference learning, semantic fact extraction,
    and proper conversation structure mapping.
    """

    def __init__(self, type: str, config: BedrockAgentCoreConfig):
        super().__init__()

        self._validate_type(type)
        self.memory_type = type
        self.config = config

        # Initialize boto3 clients
        self.bedrock_agentcore_memory_client = None

        # Initialize memory clients
        self._initialize_memory_client()

        logger.info(
            "AgentCoreStorage initialized: memory_type=%s, namespaces=%s",
            self.memory_type,
            self.config.namespaces,
        )

    def _validate_type(self, type: str) -> None:
        """Validate the memory type."""
        supported_types = {"external"}
        if type not in supported_types:
            raise ValueError(
                f"Invalid type '{type}' for AgentCoreStorage. "
                f"Must be one of: {', '.join(supported_types)}"
            )

    def _initialize_memory_client(self) -> None:
        """Initialize the Bedrock AgentCore Memory client."""
        try:
            import boto3
        except ImportError:
            raise ModuleNotFoundError(
                "boto3 is required for Bedrock AgentCore. Use `pip install crewai[agentcore]` to install the required dependencies."
            )

        try:
            # Initialize boto3 clients directly as used by the MemoryClient
            self.bedrock_agentcore_memory_client = boto3.client(
                "bedrock-agentcore", region_name=self.config.region_name
            )

        except Exception as e:
            raise ValueError(
                f"Failed to initialize Bedrock AgentCore clients: {str(e)}"
            )

    @staticmethod
    def resolve_namespace_template(
        namespace_template: str, actor_id: str, session_id: str, strategy_id: str
    ) -> str:
        """
        Resolve namespace template variables with actual values.

        This utility method handles the substitution of template variables in namespace strings:
        - {memoryStrategyId} -> strategy ID from config
        - {actorId} -> actor ID from config
        - {sessionId} -> session ID from config

        Args:
            namespace_template: The namespace template string with variables
            actor_id: The actor ID to substitute
            session_id: The session ID to substitute
            strategy_id: The strategy ID to substitute

        Returns:
            Resolved namespace string with variables substituted

        Examples:
            AgentCoreStorage.resolve_namespace_template(
                "/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}",
                "user-456", "session-789", "strategy-123"
            )
            -> "/strategies/strategy-123/actors/user-456/sessions/session-789"
        """
        substitution_vars = {
            "actorId": actor_id,
            "sessionId": session_id,
            "memoryStrategyId": strategy_id,
        }

        resolved_namespace = namespace_template.format(**substitution_vars)
        logger.debug(
            "Resolved namespace template '%s' -> '%s'",
            namespace_template,
            resolved_namespace,
        )

        return resolved_namespace

    def save(self, value: Any, metadata: Dict[str, Any]) -> None:
        """
        Save memory item to AWS Bedrock AgentCore.

        Args:
            value: The content to save (task output, conversation, etc.) (required)
            metadata: Additional metadata for the memory item (default: None)
        """

        if self.bedrock_agentcore_memory_client is None:
            raise RuntimeError("AgentCore memory client not initialized")

        payload = self._convert_to_event_payload(value, metadata)

        try:
            from datetime import datetime

            # Use the boto3 client create_event method directly
            response = self.bedrock_agentcore_memory_client.create_event(
                memoryId=str(self.config.memory_id),
                actorId=str(self.config.actor_id),
                sessionId=str(self.config.session_id),
                eventTimestamp=datetime.now(),
                payload=payload,
            )

            logger.info(
                "Created event in AgentCore memory: %s", response["event"]["eventId"]
            )

        except Exception as e:
            logger.error("Error saving to AgentCore: %s", str(e))
            raise

    def search(
        self, query: str, limit: int = 3, score_threshold: float = 0.35
    ) -> List[Dict[str, Any]]:
        """
        Search for memories using AgentCore boto3 client.

        Args:
            query: Search query (required)
            limit: Maximum number of results (default: 3, max: 100)
            score_threshold: Minimum relevance score (default: 0.35)

        Returns:
            List of memory items with context, sorted by relevance score
        """
        if self.bedrock_agentcore_memory_client is None:
            raise RuntimeError("AgentCore memory client not initialized")

        # Validate limit parameter
        if limit >= 100:
            raise ValueError("Limit must be less than 100")

        try:
            # Use the configured namespaces directly
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
                try:
                    logger.debug("Searching namespace: %s", search_namespace)

                    # Use the boto3 client retrieve_memory_records method directly
                    long_term_memory = (
                        self.bedrock_agentcore_memory_client.retrieve_memory_records(
                            memoryId=str(self.config.memory_id),
                            namespace=search_namespace,
                            searchCriteria={"searchQuery": query, "topK": limit},
                        )
                    )
                    long_term_memories = long_term_memory.get(
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
                                "context": text,
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

                except Exception as e:
                    logger.warning(
                        "Failed to search namespace %s: %s", search_namespace, str(e)
                    )
                    continue

            # Sort by score (results are already limited by top_k parameter)
            long_term_memory_results.sort(key=lambda x: x.get("score"), reverse=True)

            logger.info(
                "AgentCore search returned %d results from %d namespaces",
                len(long_term_memory_results),
                len(namespaces_to_search),
            )

            # Apply limit
            final_results = long_term_memory_results[:limit]

            # If we have fewer results than requested, search short-term memory
            if len(final_results) < limit:
                logger.info(
                    "Found %d long term memory results, searching through short term memory",
                    len(final_results),
                )
                try:
                    short_term_memory = (
                        self.bedrock_agentcore_memory_client.list_events(
                            memoryId=self.config.memory_id,
                            actorId=self.config.actor_id,
                            sessionId=self.config.session_id,
                            includePayloads=True,
                            maxResults=(limit - len(final_results)),
                        )
                    )
                    for event in short_term_memory["events"]:
                        for payload_item in event["payload"]:
                            if "conversational" in payload_item:
                                text = payload_item["conversational"]["content"]["text"]
                                role = payload_item["conversational"]["role"]
                                final_results.append(
                                    {
                                        "id": event.get("eventId"),
                                        "context": text,
                                        "metadata": {
                                            "created_at": event.get("eventTimestamp"),
                                            "role": role,
                                        },
                                        "score": 1.0,  # Default score for short-term memory
                                    }
                                )
                except Exception as e:
                    logger.warning("Failed to search short term memory: %s", str(e))

            logger.info(
                "Returning %d results",
                len(final_results),
            )
            return final_results

        except Exception as e:
            logger.error("Unexpected error searching AgentCore: %s", str(e))
            return []

    def _convert_to_event_payload(
        self, value: Any, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Convert CrewAI content to enhanced event payload that utilizes metadata when available.

        Args:
            value: The content to convert
            metadata: metadata containing messages, agent, and description

        Returns:
            List of enhanced payload dictionaries for create_event
        """
        messages = metadata["messages"]
        payload = []

        # Role mapping dictionary
        ROLE_MAPPING = {
            "SYSTEM": "OTHER",  # AWS Bedrock AgentCore doesn't recognize SYSTEM
            "USER": "USER",
            "ASSISTANT": "ASSISTANT",
            "TOOL": "TOOL",
        }

        # Process each message in the conversation
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]

            payload_role = ROLE_MAPPING.get(role, "OTHER")  # Default to OTHER

            payload.append(
                {
                    "conversational": {
                        "content": {"text": content},
                        "role": payload_role,
                    }
                }
            )

        payload.append(
            {
                "conversational": {
                    "content": {"text": str(value)},
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
