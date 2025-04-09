import pytest

from crewai.llm import LLM


@pytest.mark.vcr(filter_headers=["authorization"])
def test_qwen_multimodal_content_formatting():
    """Test that multimodal content is properly formatted for Qwen models."""
    
    llm = LLM(model="sambanova/Qwen2.5-72B-Instruct", temperature=0.7)
    
    message = {"role": "user", "content": "Describe this image"}
    formatted = llm._format_messages_for_provider([message])
    assert isinstance(formatted[0]["content"], list)
    assert formatted[0]["content"][0]["type"] == "text"
    assert formatted[0]["content"][0]["text"] == "Describe this image"
    
    multimodal_content = [
        {"type": "text", "text": "What's in this image?"}, 
        {"type": "image_url", "image_url": "https://example.com/image.jpg"}
    ]
    message = {"role": "user", "content": multimodal_content}
    formatted = llm._format_messages_for_provider([message])
    assert formatted[0]["content"] == multimodal_content
    
    messages = [
        {"role": "system", "content": "You are a visual analysis assistant."},
        {"role": "user", "content": multimodal_content}
    ]
    formatted = llm._format_messages_for_provider(messages)
    assert isinstance(formatted[0]["content"], list)
    assert formatted[1]["content"] == multimodal_content
