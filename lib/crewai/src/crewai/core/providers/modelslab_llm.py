"""
ModelsLab LLM Provider for CrewAI

A comprehensive custom LLM implementation that integrates ModelsLab's multi-modal API
capabilities with CrewAI's agent framework, enabling agents to generate text, images,
videos, and audio content during task execution.
"""

import json
import time
import requests
from typing import Any, Dict, List, Optional, Union
from crewai import BaseLLM


class ModelsLabLLM(BaseLLM):
    """
    ModelsLab LLM implementation for CrewAI agents.
    
    Provides comprehensive multi-modal AI capabilities including:
    - Text/Chat generation via LLM endpoints  
    - Image generation (text-to-image, image-to-image)
    - Video generation (text-to-video, image-to-video)
    - Audio generation (text-to-speech, voice cloning)
    - 3D content generation
    
    Example:
        ```python
        from crewai import Agent, Task, Crew
        from modelslab_llm import ModelsLabLLM
        
        # Initialize ModelsLab LLM
        llm = ModelsLabLLM(
            api_key="your_modelslab_api_key",
            model="gpt-4",  # For text generation
            temperature=0.7
        )
        
        # Create agent with multi-modal capabilities
        agent = Agent(
            role="Creative Assistant",
            goal="Create multimedia content",
            backstory="An AI assistant capable of generating text, images, videos, and audio",
            llm=llm
        )
        ```
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        temperature: Optional[float] = 0.7,
        max_tokens: Optional[int] = None,
        base_url: str = "https://modelslab.com/api/v6",
        timeout: int = 120,
        enable_multimodal: bool = True,
        **kwargs
    ):
        """
        Initialize ModelsLab LLM.
        
        Args:
            api_key: ModelsLab API key
            model: Model name for text generation (default: "gpt-4")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            base_url: ModelsLab API base URL
            timeout: Request timeout in seconds
            enable_multimodal: Enable image/video/audio generation capabilities
            **kwargs: Additional parameters
        """
        # Required: Call parent constructor with model and temperature
        super().__init__(model=model, temperature=temperature)
        
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.enable_multimodal = enable_multimodal
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        # Validate API key
        if not api_key:
            raise ValueError("ModelsLab API key is required")
    
    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
    ) -> Union[str, Any]:
        """
        Generate response using ModelsLab API.
        
        Supports both text generation and multi-modal content creation based on
        the conversation context and available tools.
        
        Args:
            messages: Input messages (string or OpenAI chat format)
            tools: Available tools/functions
            callbacks: Callback functions (not used)
            available_functions: Function implementations for tool calling
            
        Returns:
            Generated response as string
        """
        try:
            # Convert string input to message format
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            # Detect if user is requesting multi-modal content
            if self.enable_multimodal:
                multimodal_response = self._check_multimodal_request(messages)
                if multimodal_response:
                    return multimodal_response
            
            # Handle function calling if tools are provided
            if tools and available_functions:
                return self._handle_function_calling(messages, tools, available_functions)
            
            # Standard text generation
            return self._generate_text(messages)
            
        except requests.Timeout:
            raise TimeoutError(f"ModelsLab API request timed out after {self.timeout} seconds")
        except requests.RequestException as e:
            raise RuntimeError(f"ModelsLab API request failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"ModelsLab LLM error: {str(e)}")
    
    def _generate_text(self, messages: List[Dict[str, str]]) -> str:
        """Generate text response using ModelsLab chat API."""
        
        # Use uncensored chat API (OpenAI-compatible)
        url = f"{self.base_url}/uncensored_chat"
        
        payload = {
            "key": self.api_key,
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        
        if "error" in result:
            raise RuntimeError(f"ModelsLab API error: {result['error']}")
        
        if "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        
        raise ValueError("Invalid response format from ModelsLab API")
    
    def _check_multimodal_request(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """
        Check if the conversation contains requests for multi-modal content.
        
        Analyzes the latest message for keywords indicating image, video, or audio
        generation requests and routes to appropriate ModelsLab endpoints.
        """
        if not messages:
            return None
            
        latest_message = messages[-1].get("content", "").lower()
        
        # Image generation keywords
        image_keywords = [
            "generate image", "create image", "make image", "draw", "picture",
            "illustration", "artwork", "render", "visualize", "show me"
        ]
        
        # Video generation keywords  
        video_keywords = [
            "generate video", "create video", "make video", "animate", "video",
            "motion", "clip", "movie", "film"
        ]
        
        # Audio generation keywords
        audio_keywords = [
            "generate audio", "create audio", "text to speech", "voice",
            "speak", "say", "audio", "sound", "tts"
        ]
        
        # Check for multi-modal requests
        if any(keyword in latest_message for keyword in image_keywords):
            return self._generate_image(latest_message)
        elif any(keyword in latest_message for keyword in video_keywords):
            return self._generate_video(latest_message)
        elif any(keyword in latest_message for keyword in audio_keywords):
            return self._generate_audio(latest_message)
            
        return None
    
    def _generate_image(self, prompt: str) -> str:
        """Generate image using ModelsLab text-to-image API."""
        
        url = f"{self.base_url}/images/text2img"
        
        payload = {
            "key": self.api_key,
            "model_id": "flux",  # Default model
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "width": 1024,
            "height": 1024,
            "samples": 1,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
            "enhance_prompt": "yes",
            "safety_checker": "yes",
            "webhook": None,
            "track_id": None
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") == "success":
            image_url = result.get("output", [None])[0]
            if image_url:
                return f"I've generated an image for you: {image_url}\n\nThe image shows: {prompt}"
        elif result.get("status") == "processing":
            # Handle async processing
            fetch_url = result.get("fetch_result")
            if fetch_url:
                return self._poll_async_result(fetch_url, "image")
        
        raise RuntimeError(f"Image generation failed: {result.get('message', 'Unknown error')}")
    
    def _generate_video(self, prompt: str) -> str:
        """Generate video using ModelsLab text-to-video API."""
        
        url = f"{self.base_url}/video/text2video"
        
        payload = {
            "key": self.api_key,
            "model_id": "zeroscope",  # Default model
            "prompt": prompt,
            "negative_prompt": "low quality, blurry, pixelated",
            "height": 320,
            "width": 576,
            "num_frames": 24,
            "num_inference_steps": 20,
            "guidance_scale": 9.0,
            "webhook": None,
            "track_id": None
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") == "success":
            video_url = result.get("output", [None])[0] 
            if video_url:
                return f"I've generated a video for you: {video_url}\n\nThe video shows: {prompt}"
        elif result.get("status") == "processing":
            # Handle async processing
            fetch_url = result.get("fetch_result")
            if fetch_url:
                return self._poll_async_result(fetch_url, "video")
        
        raise RuntimeError(f"Video generation failed: {result.get('message', 'Unknown error')}")
    
    def _generate_audio(self, text: str) -> str:
        """Generate audio using ModelsLab text-to-speech API."""
        
        url = f"{self.base_url}/tts"
        
        payload = {
            "key": self.api_key,
            "voice_id": "21m00Tcm4TlvDq8ikWAM",  # Default voice
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "style": 0.0,
                "use_speaker_boost": True
            }
        }
        
        response = requests.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") == "success":
            audio_url = result.get("output")
            if audio_url:
                return f"I've generated audio for you: {audio_url}\n\nAudio content: {text}"
        
        raise RuntimeError(f"Audio generation failed: {result.get('message', 'Unknown error')}")
    
    def _poll_async_result(self, fetch_url: str, content_type: str) -> str:
        """Poll async result from ModelsLab API."""
        
        max_attempts = 30  # 5 minutes with 10 second intervals
        attempt = 0
        
        while attempt < max_attempts:
            try:
                response = requests.post(fetch_url, timeout=self.timeout)
                response.raise_for_status()
                
                result = response.json()
                
                if result.get("status") == "success":
                    output_url = result.get("output")
                    if isinstance(output_url, list) and output_url:
                        output_url = output_url[0]
                    
                    if output_url:
                        return f"I've generated {content_type} content for you: {output_url}"
                
                elif result.get("status") == "failed":
                    raise RuntimeError(f"{content_type.title()} generation failed: {result.get('message', 'Unknown error')}")
                
                # Still processing, wait and retry
                time.sleep(10)
                attempt += 1
                
            except Exception as e:
                if attempt >= max_attempts - 1:
                    raise RuntimeError(f"Failed to fetch {content_type} result: {str(e)}")
                time.sleep(10)
                attempt += 1
        
        raise RuntimeError(f"{content_type.title()} generation timed out")
    
    def _handle_function_calling(
        self,
        messages: List[Dict[str, str]],
        tools: List[dict],
        available_functions: Dict[str, Any]
    ) -> str:
        """
        Handle function calling with ModelsLab models that support tools.
        
        Note: Currently uses text generation to determine function calls.
        Future versions may support native function calling.
        """
        # For now, use text generation to handle tool usage
        # This can be enhanced when ModelsLab adds native function calling support
        
        system_prompt = """You are an AI assistant with access to tools. When a user asks for something that requires a tool, respond with a JSON object containing the function call.

Available tools:
"""
        for tool in tools:
            func_name = tool.get("function", {}).get("name", "unknown")
            func_desc = tool.get("function", {}).get("description", "No description")
            system_prompt += f"- {func_name}: {func_desc}\n"

        system_prompt += """
If you need to use a tool, respond with:
{"function_call": {"name": "function_name", "arguments": {"param": "value"}}}

Otherwise, respond normally.
"""
        
        # Add system prompt
        messages_with_system = [{"role": "system", "content": system_prompt}] + messages
        
        # Get response
        response = self._generate_text(messages_with_system)
        
        # Try to parse function call
        try:
            if response.strip().startswith("{") and "function_call" in response:
                func_call = json.loads(response)["function_call"]
                func_name = func_call["name"]
                func_args = func_call["arguments"]
                
                if func_name in available_functions:
                    result = available_functions[func_name](**func_args)
                    
                    # Add function result to conversation and get final response
                    messages.extend([
                        {"role": "assistant", "content": f"I'll use the {func_name} function."},
                        {"role": "function", "name": func_name, "content": str(result)}
                    ])
                    
                    return self._generate_text(messages)
        except (json.JSONDecodeError, KeyError):
            pass
        
        return response
    
    def supports_function_calling(self) -> bool:
        """Return True if the LLM supports function calling."""
        return True
    
    def supports_stop_words(self) -> bool:
        """Return True if the LLM supports stop sequences.""" 
        return True
    
    def get_context_window_size(self) -> int:
        """Return the context window size of the model."""
        # ModelsLab chat models typically support 8K-32K context
        model_contexts = {
            "gpt-4": 32000,
            "gpt-3.5-turbo": 16000,
            "claude-3": 200000,
            "claude-2": 100000,
        }
        return model_contexts.get(self.model, 8192)
    
    def __repr__(self) -> str:
        return f"ModelsLabLLM(model='{self.model}', temperature={self.temperature}, multimodal={self.enable_multimodal})"


# Convenience functions for specific content types

def create_modelslab_chat_llm(api_key: str, model: str = "gpt-4", **kwargs) -> ModelsLabLLM:
    """Create a ModelsLab LLM optimized for text/chat generation only."""
    return ModelsLabLLM(api_key=api_key, model=model, enable_multimodal=False, **kwargs)


def create_modelslab_multimodal_llm(api_key: str, model: str = "gpt-4", **kwargs) -> ModelsLabLLM:
    """Create a ModelsLab LLM with full multi-modal capabilities."""
    return ModelsLabLLM(api_key=api_key, model=model, enable_multimodal=True, **kwargs)


# Export main class
__all__ = ["ModelsLabLLM", "create_modelslab_chat_llm", "create_modelslab_multimodal_llm"]