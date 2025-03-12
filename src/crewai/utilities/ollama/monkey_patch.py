"""
Monkey patch for litellm.completion to enable local Ollama LLM usage.

This module provides a monkey patch for litellm.completion that allows CrewAI
to work with local Ollama LLM instances without requiring an OpenAI API key.
"""

import json
import logging
import requests
from types import SimpleNamespace
from typing import Dict, Any, List, Generator, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def query_ollama(
    prompt: str, 
    model: str = "llama3", 
    base_url: str = "http://localhost:11434", 
    stream: bool = False, 
    temperature: float = 0.7,
    stop: Optional[List[str]] = None
) -> Union[str, Generator]:
    """
    Query Ollama API directly
    
    Args:
        prompt: The prompt to send to Ollama
        model: The model to use (default: llama3)
        base_url: The base URL for Ollama API (default: http://localhost:11434)
        stream: Whether to stream the response (default: False)
        temperature: Temperature parameter for generation (default: 0.7)
        stop: Optional list of stop sequences
        
    Returns:
        The response text from Ollama or a generator for streaming
    """
    url = f"{base_url}/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": 100,
            "stream": stream
        }
    }
    
    # Add stop sequences if provided
    if stop and isinstance(stop, list) and len(stop) > 0:
        data["options"]["stop"] = stop
    
    try:
        if stream:
            # For streaming, return a generator
            response = requests.post(url, json=data, stream=True)
            response.raise_for_status()
            
            def stream_generator():
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            yield chunk["response"]
                        if chunk.get("done", False):
                            break
            return stream_generator()
        else:
            # For non-streaming, return the complete response
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json().get("response", "")
    except Exception as e:
        logger.error(f"Error querying Ollama API: {str(e)}")
        return f"Error: {str(e)}"

def extract_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    """
    Extract a prompt from a list of messages
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        A formatted prompt string
    """
    prompt = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role and content:
            prompt += f"### {role.capitalize()}:\n{content}\n\n"
    return prompt

def apply_monkey_patch() -> bool:
    """
    Apply the monkey patch to litellm.completion
    
    This function saves the original litellm.completion function and
    replaces it with a custom implementation that handles Ollama models.
    
    Returns:
        bool: True if the patch was applied successfully, False otherwise
    """
    try:
        # Import litellm
        import litellm
        logger.info("Successfully imported litellm")
        
        # Save the original completion function
        original_completion = litellm.completion
        logger.info("Saved original litellm.completion function")
        
        # Define the monkey patch function
        def custom_completion(*args, **kwargs):
            """Custom implementation of litellm.completion for Ollama"""
            model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])
            temperature = kwargs.get("temperature", 0.7)
            stream = kwargs.get("stream", False)
            base_url = kwargs.get("base_url", "http://localhost:11434")
            stop = kwargs.get("stop", None)
            
            logger.debug(f"Intercepted call to litellm.completion with model: {model}")
            
            # Only intercept calls for Ollama models
            if not model.startswith("ollama/"):
                logger.debug("Not an Ollama model, calling original litellm.completion")
                return original_completion(*args, **kwargs)
            
            # Extract the actual model name from the 'ollama/model' format
            ollama_model = model.split("/")[1]
            logger.info(f"Handling Ollama model: {ollama_model}")
            
            # Extract prompt from messages
            prompt = extract_prompt_from_messages(messages)
            
            logger.debug(f"Generated prompt: {prompt[:100]}...")
            
            # Query Ollama
            if stream:
                logger.debug("Using streaming mode")
                # For streaming, return a generator that yields chunks in the format expected by CrewAI
                # First, get the generator from query_ollama
                chunks_generator = query_ollama(
                    prompt, 
                    model=ollama_model, 
                    base_url=base_url, 
                    stream=True, 
                    temperature=temperature,
                    stop=stop
                )
                
                # Then create a wrapper generator that transforms the chunks
                def stream_response():
                    for chunk in chunks_generator:
                        yield SimpleNamespace(
                            choices=[
                                SimpleNamespace(
                                    delta=SimpleNamespace(
                                        content=chunk,
                                        role="assistant"
                                    ),
                                    index=0,
                                    finish_reason=None
                                )
                            ],
                            usage=None,
                            model=model
                        )
                    # Final chunk with finish_reason and usage
                    yield SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(
                                    content="",
                                    role="assistant"
                                ),
                                index=0,
                                finish_reason="stop"
                            )
                        ],
                        usage=SimpleNamespace(
                            prompt_tokens=len(prompt.split()),
                            completion_tokens=len(prompt.split()) * 2,  # Estimate
                            total_tokens=len(prompt.split()) * 3  # Estimate
                        ),
                        model=model
                    )
                return stream_response()
            else:
                logger.debug("Using non-streaming mode")
                # For non-streaming, return a complete response object
                response_text = query_ollama(
                    prompt, 
                    model=ollama_model, 
                    base_url=base_url, 
                    temperature=temperature,
                    stop=stop
                )
                logger.debug(f"Received response: {response_text[:100]}...")
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=response_text,
                                tool_calls=None,
                                role="assistant"
                            ),
                            finish_reason="stop",
                            index=0
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=len(prompt.split()),
                        completion_tokens=len(response_text.split()),
                        total_tokens=len(prompt.split()) + len(response_text.split())
                    ),
                    id="ollama-response",
                    model=model,
                    created=123456789
                )
        
        # Apply the monkey patch
        litellm.completion = custom_completion
        logger.info("Applied monkey patch to litellm.completion")
        
        return True
    except ImportError as e:
        logger.error(f"Error importing litellm: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
