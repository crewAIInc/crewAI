import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from bedrock_agentcore.memory.client import MemoryClient
from crewai.memory.storage.interface import Storage

# Setup logger
logger = logging.getLogger(__name__)


class BedrockAgentCoreStrategyConfig(BaseModel):
    """Configuration for an AgentCore memory strategy."""

    name: str = Field(..., description="Name of the strategy (required)")
    namespaces: List[str] = Field(
        ..., description="List of namespace templates for the strategy (required)"
    )
    strategy_id: str = Field(
        ..., description="Memory strategy ID from AWS Bedrock Agent Core"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValueError("Strategy name cannot be empty")
        return v.strip()

    @field_validator("namespaces")
    @classmethod
    def validate_namespaces(cls, v):
        if not v or not isinstance(v, list):
            raise ValueError("Strategy namespaces must be a non-empty list")

        validated_namespaces = []
        for namespace in v:
            if not namespace or not namespace.strip():
                raise ValueError("Namespace cannot be empty")
            validated_namespaces.append(namespace.strip())

        return validated_namespaces


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
    run_id: Optional[str] = Field(
        default=None, description="CrewAI run ID (default: None)"
    )

    # Strategy configuration
    strategies: List[BedrockAgentCoreStrategyConfig] = Field(
        default_factory=list,
        description="List of memory strategies with their namespaces (default: empty list)",
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

    @field_validator("strategies")
    @classmethod
    def validate_strategies(cls, v):
        if not isinstance(v, list):
            raise ValueError("Strategies must be a list")

        # Check for duplicate strategy names
        names = [strategy.name for strategy in v]
        if len(names) != len(set(names)):
            raise ValueError("Strategy names must be unique")

        return v

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

        # Initialize memory client to None first
        self.memory_client = None

        # Initialize strategy namespace mapping with strong typing
        self.strategy_namespace_map: Dict[str, List[str]] = {}

        # Setup strategy namespaces
        self._setup_strategy_namespaces()

        # Initialize memory client
        self._initialize_memory_client()

        logger.info(
            "AgentCoreStorage initialized: memory_type=%s, run_id=%s",
            self.memory_type,
            self.config.run_id,
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
        if MemoryClient is None:
            raise RuntimeError("bedrock-agentcore-sdk-python is not available")
        try:
            # Initialize the MemoryClient from bedrock-agentcore-sdk-python
            self.memory_client = MemoryClient(region_name=self.config.region_name)

        except Exception as e:
            raise ValueError(
                f"Failed to initialize Bedrock AgentCore Memory client: {str(e)}"
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

    def _setup_strategy_namespaces(self) -> None:
        """
        Setup namespace mappings for each strategy.

        Note: Namespaces should already be resolved by the user before creating AgentCoreConfig.
        This method simply stores the provided namespaces.
        """
        self.strategy_namespace_map = {}

        if not self.config.strategies:
            logger.debug("No strategies configured")
            return

        for strategy_config in self.config.strategies:
            # Store the namespaces as provided (should already be resolved)
            self.strategy_namespace_map[strategy_config.name] = (
                strategy_config.namespaces
            )
            logger.debug(
                "Strategy '%s' namespaces: %s",
                strategy_config.name,
                strategy_config.namespaces,
            )

    def get_namespaces_for_strategy(self, strategy: str) -> Optional[List[str]]:
        """
        Get the namespaces for a specific strategy.

        Args:
            strategy: The strategy name

        Returns:
            List of namespaces for the strategy, or None if strategy not found
        """
        return self.strategy_namespace_map.get(strategy)

    def get_all_unique_namespaces(self) -> List[str]:
        """
        Get all unique namespaces from all configured strategies.

        Returns:
            List of unique namespaces across all strategies
        """
        all_namespaces = []
        for strategy_namespaces in self.strategy_namespace_map.values():
            all_namespaces.extend(strategy_namespaces)

        return list(set(all_namespaces))

    def save(self, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save memory item to AWS Bedrock AgentCore.

        Args:
            value: The content to save (task output, conversation, etc.) (required)
            metadata: Additional metadata for the memory item (default: None)
        """

        if self.memory_client is None:
            raise RuntimeError("AgentCore memory client not initialized")

        # Convert CrewAI content to message tuples for MemoryClient
        message_tuples = self._convert_to_message_tuples(value, metadata)

        try:
            # Use the MemoryClient create_event method
            response = self.memory_client.create_event(
                memory_id=str(self.config.memory_id),
                actor_id=str(self.config.actor_id),
                session_id=str(self.config.session_id),
                messages=message_tuples,
            )

            event_id = response.get("eventId", "Unknown")
            logger.info("Created event in AgentCore memory: %s", event_id)

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
            limit: Maximum number of results (default: 3)
            score_threshold: Minimum relevance score (default: 0.35)

        Returns:
            List of memory items with context, sorted by relevance score
        """
        if self.memory_client is None:
            raise RuntimeError("AgentCore memory client not initialized")

        try:
            # Determine which namespaces to search - use all configured namespaces
            if self.strategy_namespace_map:
                # Get all unique namespaces from all strategies
                namespaces_to_search = self.get_all_unique_namespaces()
                logger.debug(
                    "Searching all configured strategy namespaces: %s",
                    namespaces_to_search,
                )
            else:
                logger.warning("No strategies configured - no namespaces to search")
                return []

            all_results = []

            # Search each namespace
            for search_namespace in namespaces_to_search:
                try:
                    logger.debug("Searching namespace: %s", search_namespace)

                    memories = self.memory_client.retrieve_memories(
                        memory_id=str(self.config.memory_id),
                        namespace=search_namespace,
                        query=query,
                        top_k=limit,
                    )

                    # Process search results
                    if memories and isinstance(memories, list):
                        for memory in memories:
                            # Extract text content from memory record
                            content = memory.get("content", {})
                            text = content.get("text", "")

                            # Get score if available
                            score = memory.get("score", 1.0)

                            # Apply score threshold filter
                            if score < score_threshold:
                                continue

                            all_results.append(
                                {
                                    "id": memory.get("memoryRecordId"),
                                    "context": text,
                                    "metadata": {
                                        "namespaces": memory.get("namespaces"),
                                        "created_at": memory.get("createdAt"),
                                        "search_namespace": search_namespace,
                                        "memory_strategy_id": memory.get(
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

            # Sort by score and limit results
            all_results.sort(key=lambda x: x.get("score"), reverse=True)
            final_results = all_results[:limit]

            logger.info(
                "AgentCore search returned %d results from %d namespaces",
                len(final_results),
                len(namespaces_to_search),
            )
            return final_results

        except Exception as e:
            logger.error("Unexpected error searching AgentCore: %s", str(e))
            return []

    def _convert_to_message_tuples(
        self, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Convert CrewAI content to message tuples for the MemoryClient.

        Args:
            value: The content to convert - can be a single message or a list of message dictionaries
            metadata: Additional metadata (used for single messages)

        Returns:
            List of (text, role) tuples

        Raises:
            ValueError: If more than 100 messages are provided in a batch
        """
        # Handle list of message dictionaries (batch processing)
        if isinstance(value, list):
            # Validate batch size limit
            if len(value) > 100:
                raise ValueError(
                    f"Cannot process more than 100 messages in a batch. "
                    f"Received {len(value)} messages. Please split into smaller batches."
                )

            message_tuples = []
            for message_item in value:
                if isinstance(message_item, dict) and "content" in message_item:
                    # Extract content and metadata from each message
                    content = message_item["content"]
                    msg_metadata = message_item.get("metadata", {})

                    # Determine role from message metadata
                    role = self._determine_role_from_metadata(msg_metadata)

                    # Convert content to string
                    if isinstance(content, str):
                        content_text = content
                    elif hasattr(content, "raw"):
                        content_text = str(content.raw)
                    elif isinstance(content, dict):
                        content_text = json.dumps(content, default=str)
                    else:
                        content_text = str(content)

                    message_tuples.append((content_text, role))
                else:
                    # Handle non-dict items in the list
                    role = self._determine_role_from_metadata(metadata or {})
                    content_text = str(message_item)
                    message_tuples.append((content_text, role))

            return message_tuples

        # Handle single message (existing behavior)
        role = self._determine_role_from_metadata(metadata or {})

        # Handle different value types
        if isinstance(value, str):
            content_text = value
        elif hasattr(value, "raw"):
            # CrewAI TaskOutput or similar
            content_text = str(value.raw)
        elif isinstance(value, dict):
            content_text = json.dumps(value, default=str)
        else:
            content_text = str(value)

        # Return as list of tuples for MemoryClient
        return [(content_text, role)]

    def _determine_role_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Determine the message role from metadata.

        Args:
            metadata: Message metadata dictionary

        Returns:
            Role string ("USER", "ASSISTANT", or "TOOL")
        """
        if not metadata:
            return "ASSISTANT"  # Default role

        # Check explicit role in metadata
        if "role" in metadata:
            role_value = str(metadata["role"]).upper()
            if role_value in ["USER", "ASSISTANT", "TOOL"]:
                return role_value

        # Check if this is from a specific agent or user
        if metadata.get("agent"):
            return "ASSISTANT"
        elif "user" in str(metadata).lower():
            return "USER"
        elif "tool" in str(metadata).lower():
            return "TOOL"

        return "ASSISTANT"  # Default role

    def reset(self) -> None:
        """Reset/clear memory storage."""
        logger.warning(
            "Bedrock AgentCore memory reset not implemented - memories persist in AWS"
        )
        # Note: Bedrock AgentCore doesn't provide a direct reset API
        # delete_memory can be used to delete the entire resource
