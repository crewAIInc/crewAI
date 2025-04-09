class OllamaConnectionException(Exception):
    """Exception raised when there's a connection issue with Ollama.
    
    This typically happens when Ollama is not running or is not accessible
    at the expected URL.
    """
    
    def __init__(self, error_message: str):
        self.original_error_message = error_message
        super().__init__(self._get_error_message(error_message))
    
    def _get_error_message(self, error_message: str):
        return (
            f"Failed to connect to Ollama. Original error: {error_message}\n"
            "Please make sure Ollama is installed and running. "
            "You can install Ollama from https://ollama.com/download and "
            "start it by running 'ollama serve' in your terminal."
        )
