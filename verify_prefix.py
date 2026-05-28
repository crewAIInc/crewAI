import sys
import os

# This forces Python to look in your local source folder for the 'crewai' package
sys.path.insert(0, os.path.abspath("lib/crewai/src"))

# Now your import will work
from crewai.llm import LLM

# Simulate the environment variable
os.environ["CREWAI_LLM_PREFIXES"] = "my-custom-model-"

# Test the function directly
is_match = LLM._matches_provider_pattern("my-custom-model-v1", "openai")

if is_match:
    print("✅ SUCCESS: The custom prefix was detected!")
else:
    print("❌ FAILED: The custom prefix was not detected.")