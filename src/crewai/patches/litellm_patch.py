"""
Patch for litellm to fix IndexError in ollama_pt function.

This patch addresses issue #2744 in the crewAI repository, where an IndexError occurs
in litellm's Ollama prompt template function when CrewAI Agent with Tools uses Ollama/Qwen models.

Version: 1.0.0
"""

import json
import logging
from typing import Any, Dict, List, Union, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Patch version
PATCH_VERSION = "1.0.0"


class PatchApplicationError(Exception):
    """Exception raised when a patch fails to apply."""
    pass


def apply_patches() -> bool:
    """
    Apply all patches to fix known issues with dependencies.
    
    Returns:
        bool: True if all patches were applied successfully, False otherwise.
    """
    success = patch_litellm_ollama_pt()
    logger.info(f"LiteLLM ollama_pt patch applied: {success}")
    return success


def patch_litellm_ollama_pt() -> bool:
    """
    Patch the ollama_pt function in litellm to fix IndexError.
    
    The issue occurs when accessing messages[msg_i].get("tool_calls") without checking
    if msg_i is within bounds of the messages list. This happens after tool execution
    during the next LLM call.
    
    Returns:
        bool: True if the patch was applied successfully, False otherwise.
    
    Raises:
        PatchApplicationError: If there's an error during patch application.
    """
    try:
        # Import the module containing the function to patch
        import litellm.litellm_core_utils.prompt_templates.factory as factory
        
        # Define a patched version of the function
        def patched_ollama_pt(model: str, messages: List[Dict]) -> Dict[str, Any]:
            """
            Patched version of ollama_pt that adds bounds checking.
            
            This fixes the IndexError that occurs when the assistant message is the last
            message in the list and msg_i goes out of bounds.
            
            Args:
                model: The model name.
                messages: The list of messages to process.
                
            Returns:
                Dict containing the prompt and images.
            """
            user_message_types = {"user", "tool", "function"}
            msg_i = 0
            images: List[str] = []
            prompt = ""
            
            # Handle empty messages list
            if not messages:
                return {"prompt": prompt, "images": images}
                
            while msg_i < len(messages):
                init_msg_i = msg_i
                user_content_str = ""
                ## MERGE CONSECUTIVE USER CONTENT ##
                while msg_i < len(messages) and messages[msg_i]["role"] in user_message_types:
                    msg_content = messages[msg_i].get("content")
                    if msg_content:
                        if isinstance(msg_content, list):
                            for m in msg_content:
                                if m.get("type", "") == "image_url":
                                    if isinstance(m["image_url"], str):
                                        images.append(m["image_url"])
                                    elif isinstance(m["image_url"], dict):
                                        images.append(m["image_url"]["url"])
                                elif m.get("type", "") == "text":
                                    user_content_str += m["text"]
                        else:
                            # Tool message content will always be a string
                            user_content_str += msg_content

                    msg_i += 1

                if user_content_str:
                    prompt += f"### User:\n{user_content_str}\n\n"

                system_content_str, msg_i = factory._handle_ollama_system_message(
                    messages, prompt, msg_i
                )
                if system_content_str:
                    prompt += f"### System:\n{system_content_str}\n\n"

                assistant_content_str = ""
                ## MERGE CONSECUTIVE ASSISTANT CONTENT ##
                while msg_i < len(messages) and messages[msg_i]["role"] == "assistant":
                    assistant_content_str += factory.convert_content_list_to_str(messages[msg_i])
                    msg_i += 1

                    # Add bounds check before accessing messages[msg_i]
                    # This is the key fix for the IndexError
                    if msg_i < len(messages):
                        tool_calls = messages[msg_i].get("tool_calls")
                        ollama_tool_calls = []
                        if tool_calls:
                            for call in tool_calls:
                                call_id = call["id"]
                                function_name = call["function"]["name"]
                                arguments = json.loads(call["function"]["arguments"])

                                ollama_tool_calls.append(
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": function_name,
                                            "arguments": arguments,
                                        },
                                    }
                                )

                        if ollama_tool_calls:
                            assistant_content_str += (
                                f"Tool Calls: {json.dumps(ollama_tool_calls, indent=2)}"
                            )

                            msg_i += 1

                if assistant_content_str:
                    prompt += f"### Assistant:\n{assistant_content_str}\n\n"

                if msg_i == init_msg_i:  # prevent infinite loops
                    raise factory.litellm.BadRequestError(
                        message=factory.BAD_MESSAGE_ERROR_STR + f"passed in {messages[msg_i]}",
                        model=model,
                        llm_provider="ollama",
                    )

            response_dict = {
                "prompt": prompt,
                "images": images,
            }

            return response_dict
        
        # Replace the original function with our patched version
        factory.ollama_pt = patched_ollama_pt
        
        logger.info(f"Successfully applied litellm ollama_pt patch version {PATCH_VERSION}")
        return True
    except Exception as e:
        error_msg = f"Failed to apply litellm ollama_pt patch: {e}"
        logger.error(error_msg)
        return False


# For backwards compatibility
def patch_litellm() -> bool:
    """
    Legacy function for backwards compatibility.
    
    Returns:
        bool: True if the patch was applied successfully, False otherwise.
    """
    try:
        return patch_litellm_ollama_pt()
    except Exception as e:
        logger.error(f"Failed to apply legacy litellm patch: {e}")
        return False
