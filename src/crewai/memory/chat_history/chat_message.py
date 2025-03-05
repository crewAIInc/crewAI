from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MessageRole(str, Enum):
    """Enum for message roles in a chat."""
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"


class ChatMessage:
    """
    Represents a single message in a chat history.
    
    Attributes:
        role: The role of the message sender (human, ai, or system).
        content: The content of the message.
        timestamp: When the message was created.
        metadata: Additional information about the message.
    """
    
    def __init__(
        self,
        role: MessageRole,
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the message to a dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatMessage":
        """Create a message from a dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )
