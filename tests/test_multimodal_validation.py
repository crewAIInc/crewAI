from typing import Dict, List, Union, Any
from pydantic import ValidationError
import pytest

from crewai.utilities.events.llm_events import LLMCallStartedEvent


def test_llm_call_started_event_with_multimodal_content():
    """Test that LLMCallStartedEvent properly handles multimodal content."""
    # Create a multimodal message structure
    multimodal_message = {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'Please analyze this image'},
            {
                'type': 'image_url',
                'image_url': {
                    'url': 'https://example.com/test-image.jpg',
                },
            },
        ],
    }
    
    # This should not raise a ValidationError
    event = LLMCallStartedEvent(messages=[multimodal_message])
    
    # Verify the event was created correctly
    assert event.messages[0]['role'] == 'user'
    assert isinstance(event.messages[0]['content'], list)
    assert len(event.messages[0]['content']) == 2
    assert event.messages[0]['content'][0]['type'] == 'text'
    assert event.messages[0]['content'][1]['type'] == 'image_url'


def test_llm_call_started_event_with_string_message():
    """Test that LLMCallStartedEvent still works with string messages."""
    # Create a simple string message
    message = "This is a test message"
    
    # This should not raise a ValidationError
    event = LLMCallStartedEvent(messages=message)
    
    # Verify the event was created correctly
    assert event.messages == message


def test_llm_call_started_event_with_standard_messages():
    """Test that LLMCallStartedEvent still works with standard message format."""
    # Create standard messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    # This should not raise a ValidationError
    event = LLMCallStartedEvent(messages=messages)
    
    # Verify the event was created correctly
    assert len(event.messages) == 2
    assert event.messages[0]['role'] == 'system'
    assert event.messages[0]['content'] == 'You are a helpful assistant'
    assert event.messages[1]['role'] == 'user'
    assert event.messages[1]['content'] == 'Hello, how are you?'
