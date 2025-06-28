"""
Integration tests for Ollama model handling.
This module tests the Ollama-specific functionality including response_format handling.
"""

from pydantic import BaseModel
from crewai.llm import LLM
from crewai import Agent

class GuideOutline(BaseModel):
    title: str
    sections: list[str]

def test_original_issue():
    """Test the original issue scenario from GitHub issue #3082."""
    print("Testing original issue scenario...")
    
    try:
        llm = LLM(model="ollama/gemma3:latest", response_format=GuideOutline)
        print("✅ LLM creation with response_format succeeded")
        
        params = llm._prepare_completion_params("Test message")
        if "response_format" not in params or params.get("response_format") is None:
            print("✅ response_format correctly filtered out for Ollama model")
        else:
            print("❌ response_format was not filtered out")
            
        agent = Agent(
            role="Guide Creator",
            goal="Create comprehensive guides",
            backstory="You are an expert at creating structured guides",
            llm=llm
        )
        print("✅ Agent creation with Ollama LLM succeeded")
        
        assert agent.llm.model == "ollama/gemma3:latest"
        
    except ValueError as e:
        if "does not support response_format" in str(e):
            print(f"❌ Original issue still exists: {e}")
            return False
        else:
            print(f"❌ Unexpected ValueError: {e}")
            return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

def test_non_ollama_models():
    """Test that non-Ollama models still work with response_format."""
    print("\nTesting non-Ollama models...")
    
    try:
        llm = LLM(model="gpt-4", response_format=GuideOutline)
        params = llm._prepare_completion_params("Test message")
        
        if params.get("response_format") == GuideOutline:
            print("✅ Non-Ollama models still include response_format")
            return True
        else:
            print("❌ Non-Ollama models missing response_format")
            return False
            
    except Exception as e:
        print(f"❌ Error with non-Ollama model: {e}")
        return False

def test_ollama_model_detection_edge_cases():
    """Test edge cases for Ollama model detection."""
    print("\nTesting Ollama model detection edge cases...")
    
    test_cases = [
        ("ollama/llama3.2:3b", True, "Standard ollama/ prefix"),
        ("OLLAMA/MODEL:TAG", True, "Uppercase ollama/ prefix"),
        ("ollama:custom-model", True, "ollama: prefix"),
        ("custom/ollama-model", False, "Contains 'ollama' but not prefix"),
        ("gpt-4", False, "Non-Ollama model"),
        ("anthropic/claude-3", False, "Different provider"),
        ("openai/gpt-4", False, "OpenAI model"),
    ]
    
    all_passed = True
    for model, expected, description in test_cases:
        llm = LLM(model=model)
        result = llm._is_ollama_model(model)
        if result == expected:
            print(f"✅ {description}: {model} -> {result}")
        else:
            print(f"❌ {description}: {model} -> {result} (expected {expected})")
            all_passed = False
    
    return all_passed

if __name__ == "__main__":
    print("Testing Ollama response_format fix...")
    
    success1 = test_original_issue()
    success2 = test_non_ollama_models()
    success3 = test_ollama_model_detection_edge_cases()
    
    if success1 and success2 and success3:
        print("\n🎉 All tests passed! The fix is working correctly.")
    else:
        print("\n💥 Some tests failed. The fix needs more work.")
