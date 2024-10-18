ENV_VARS = {
    'openai': ['OPENAI_API_KEY'],
    'anthropic': ['ANTHROPIC_API_KEY'],
    'gemini': ['GEMINI_API_KEY'],
    'groq': ['GROQ_API_KEY'],
    'ollama': ['FAKE_KEY'],
}

PROVIDERS = ['openai', 'anthropic', 'gemini', 'groq', 'ollama']

MODELS = {
    'openai': ['gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o1-mini', 'o1-preview'],
    'anthropic': ['claude-3-5-sonnet-20240620', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
    'gemini': ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-gemma-2-9b-it', 'gemini-gemma-2-27b-it'],
    'groq': ['llama-3.1-8b-instant', 'llama-3.1-70b-versatile', 'llama-3.1-405b-reasoning', 'gemma2-9b-it', 'gemma-7b-it'],
    'ollama': ['llama3.1', 'mixtral'],
}

JSON_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"