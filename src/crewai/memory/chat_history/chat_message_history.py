from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import PrivateAttr

from crewai.memory.chat_history.chat_message import ChatMessage, MessageRole
from crewai.memory.memory import Memory
from crewai.memory.storage.rag_storage import RAGStorage


class ChatMessageHistory(Memory):
    """
    ChatMessageHistory class for storing and retrieving chat messages.
    
    This class allows for maintaining conversation context across multiple
    interactions within a single session, similar to Langchain's ChatMessageHistory.
    
    Attributes:
        messages: A list of ChatMessage objects representing the conversation history.
    """
    
    _memory_provider: Optional[str] = PrivateAttr()
    _messages: List[ChatMessage] = PrivateAttr(default_factory=list)
    
    def __init__(self, crew=None, embedder_config=None, storage=None, path=None):
        if crew and hasattr(crew, "memory_config") and crew.memory_config is not None:
            memory_provider = crew.memory_config.get("provider")
        else:
            memory_provider = None

        if memory_provider == "mem0":
            try:
                from crewai.memory.storage.mem0_storage import Mem0Storage
            except ImportError:
                raise ImportError(
                    "Mem0 is not installed. Please install it with `pip install mem0ai`."
                )
            storage = Mem0Storage(type="chat_history", crew=crew)
        else:
            storage = (
                storage
                if storage
                else RAGStorage(
                    type="chat_history",
                    embedder_config=embedder_config,
                    crew=crew,
                    path=path,
                )
            )
        super().__init__(storage=storage)
        self._memory_provider = memory_provider
        self._messages = []
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role: The role of the message sender (human, ai, or system).
            content: The content of the message.
            metadata: Additional information about the message.
            agent: The agent associated with the message.
        """
        message = ChatMessage(role=role, content=content, metadata=metadata)
        self._messages.append(message)
        
        # Save to storage for persistence and retrieval
        metadata = metadata or {}
        if agent:
            metadata["agent"] = agent
        
        # Add role and timestamp to metadata
        metadata["role"] = role.value
        metadata["timestamp"] = message.timestamp.isoformat()
        
        super().save(value=content, metadata=metadata, agent=agent)
    
    def add_human_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Add a human message to the chat history."""
        self.add_message(MessageRole.HUMAN, content, metadata, agent)
    
    def add_ai_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Add an AI message to the chat history."""
        self.add_message(MessageRole.AI, content, metadata, agent)
    
    def add_system_message(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Add a system message to the chat history."""
        self.add_message(MessageRole.SYSTEM, content, metadata, agent)
    
    def get_messages(self) -> List[ChatMessage]:
        """Get all messages in the chat history."""
        return self._messages
    
    def get_messages_as_dict(self) -> List[Dict[str, Any]]:
        """Get all messages in the chat history as dictionaries."""
        return [message.to_dict() for message in self._messages]
    
    def clear(self) -> None:
        """Clear all messages from the chat history."""
        self._messages = []
        self.reset()
    
    def reset(self) -> None:
        """Reset the storage."""
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the chat message history: {e}"
            )
    
    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Search for messages in the chat history.
        
        Args:
            query: The search query.
            limit: The maximum number of results to return.
            score_threshold: The minimum similarity score for results.
            
        Returns:
            A list of dictionaries containing the search results.
        """
        results = self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        )
        
        # Convert the search results to ChatMessage objects
        messages = []
        for result in results:
            try:
                role = result["metadata"].get("role", "ai")
                content = result["context"]
                timestamp = result["metadata"].get("timestamp")
                if timestamp:
                    timestamp = datetime.fromisoformat(timestamp)
                else:
                    timestamp = datetime.now()
                
                metadata = {k: v for k, v in result["metadata"].items() 
                           if k not in ["role", "timestamp"]}
                
                message = ChatMessage(
                    role=MessageRole(role),
                    content=content,
                    timestamp=timestamp,
                    metadata=metadata,
                )
                messages.append(message.to_dict())
            except Exception:
                # Skip invalid messages
                continue
        
        return messages
