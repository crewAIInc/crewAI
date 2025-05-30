"""Custom error classes for CrewAI."""

class ChromaDBRequiredError(ImportError):
    """Error raised when ChromaDB is required but not installed."""
    
    def __init__(self, feature: str):
        """Initialize the error with a specific feature name.
        
        Args:
            feature: The name of the feature that requires ChromaDB.
        """
        message = (
            f"ChromaDB is required for {feature} features. "
            "Please install it with 'pip install crewai[storage]'"
        )
        super().__init__(message)
