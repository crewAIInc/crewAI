import pytest
from datetime import datetime

from crewai.memory.chat_history.chat_message_history import ChatMessageHistory
from crewai.memory.chat_history.chat_message import ChatMessage, MessageRole


@pytest.fixture
def chat_message_history():
    """Fixture to create a ChatMessageHistory instance"""
    return ChatMessageHistory()


def test_add_and_get_messages(chat_message_history):
    """Test adding messages and retrieving them."""
    # Add messages
    chat_message_history.add_human_message("Hello, how are you?")
    chat_message_history.add_ai_message("I'm doing well, thank you!")
    chat_message_history.add_system_message("System message")
    
    # Get messages
    messages = chat_message_history.get_messages()
    
    # Verify messages
    assert len(messages) == 3
    assert messages[0].role == MessageRole.HUMAN
    assert messages[0].content == "Hello, how are you?"
    assert messages[1].role == MessageRole.AI
    assert messages[1].content == "I'm doing well, thank you!"
    assert messages[2].role == MessageRole.SYSTEM
    assert messages[2].content == "System message"


def test_get_messages_as_dict(chat_message_history):
    """Test getting messages as dictionaries."""
    # Add messages
    chat_message_history.add_human_message("Hello")
    chat_message_history.add_ai_message("Hi there")
    
    # Get messages as dict
    messages_dict = chat_message_history.get_messages_as_dict()
    
    # Verify messages
    assert len(messages_dict) == 2
    assert messages_dict[0]["role"] == "human"
    assert messages_dict[0]["content"] == "Hello"
    assert messages_dict[1]["role"] == "ai"
    assert messages_dict[1]["content"] == "Hi there"
    assert "timestamp" in messages_dict[0]
    assert "metadata" in messages_dict[0]


def test_clear_messages(chat_message_history):
    """Test clearing messages."""
    # Add messages
    chat_message_history.add_human_message("Hello")
    chat_message_history.add_ai_message("Hi there")
    
    # Verify messages were added
    assert len(chat_message_history.get_messages()) == 2
    
    # Clear messages
    chat_message_history.clear()
    
    # Verify messages were cleared
    assert len(chat_message_history.get_messages()) == 0


def test_search_messages(chat_message_history, monkeypatch):
    """Test searching for messages."""
    # Add messages with specific content
    chat_message_history.add_human_message(
        "I need information about machine learning algorithms"
    )
    chat_message_history.add_ai_message(
        "Machine learning algorithms include decision trees, neural networks, and SVMs"
    )
    chat_message_history.add_human_message(
        "Tell me more about neural networks"
    )
    
    # Mock storage search results
    mock_search_results = [
        {
            "context": "Machine learning algorithms include decision trees, neural networks, and SVMs",
            "metadata": {
                "role": "ai",
                "timestamp": "2023-01-01T00:00:00"
            }
        }
    ]
    
    # Monkeypatch the storage.search method
    def mock_storage_search(*args, **kwargs):
        return mock_search_results
    
    monkeypatch.setattr(chat_message_history.storage, "search", mock_storage_search)
    
    # Search for messages about neural networks
    results = chat_message_history.search("neural networks")
    
    # Verify search results
    assert len(results) > 0
    assert any("neural networks" in result["content"] for result in results)


def test_message_with_metadata(chat_message_history):
    """Test adding and retrieving messages with metadata."""
    # Add message with metadata
    metadata = {"user_id": "123", "session_id": "abc"}
    chat_message_history.add_human_message(
        "Hello with metadata", metadata=metadata
    )
    
    # Get messages
    messages = chat_message_history.get_messages()
    
    # Verify metadata
    assert len(messages) == 1
    assert messages[0].metadata["user_id"] == "123"
    assert messages[0].metadata["session_id"] == "abc"


def test_chat_message_to_from_dict():
    """Test converting ChatMessage to and from dictionary."""
    # Create a message
    timestamp = datetime.now()
    message = ChatMessage(
        role=MessageRole.HUMAN,
        content="Test message",
        timestamp=timestamp,
        metadata={"key": "value"}
    )
    
    # Convert to dict
    message_dict = message.to_dict()
    
    # Verify dict
    assert message_dict["role"] == "human"
    assert message_dict["content"] == "Test message"
    assert message_dict["timestamp"] == timestamp.isoformat()
    assert message_dict["metadata"] == {"key": "value"}
    
    # Convert back to ChatMessage
    new_message = ChatMessage.from_dict(message_dict)
    
    # Verify new message
    assert new_message.role == MessageRole.HUMAN
    assert new_message.content == "Test message"
    assert new_message.timestamp.isoformat() == timestamp.isoformat()
    assert new_message.metadata == {"key": "value"}
