# test_azure_integration.py
from src.crewai.llm import LLM

# Test with Azure parameters but without azure/ prefix
llm = LLM(
    api_key='test_key',
    api_base='test_base',
    model='gpt-4o-mini-2024-07-18',
    api_version='test_version'
)

# Print the detected provider
provider = llm._get_custom_llm_provider()
print(f"Detected provider: {provider}")
print(f"Is Azure detected correctly: {provider == 'azure'}")

# Prepare parameters that would be passed to LiteLLM
params = llm._prepare_completion_params(messages=[{"role": "user", "content": "test"}])
print(f"Parameters passed to LiteLLM: {params}")

# Test with Azure parameters and with azure/ prefix for comparison
llm_with_prefix = LLM(
    api_key='test_key',
    api_base='test_base',
    model='azure/gpt-4o-mini-2024-07-18',
    api_version='test_version'
)

# Print the detected provider
provider_with_prefix = llm_with_prefix._get_custom_llm_provider()
print(f"\nWith azure/ prefix:")
print(f"Detected provider: {provider_with_prefix}")
print(f"Is Azure detected correctly: {provider_with_prefix == 'azure'}")

# Prepare parameters that would be passed to LiteLLM
params_with_prefix = llm_with_prefix._prepare_completion_params(messages=[{"role": "user", "content": "test"}])
print(f"Parameters passed to LiteLLM: {params_with_prefix}")
