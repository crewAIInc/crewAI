class AgentExecutionTimeoutError(Exception):
    """Exception raised when an agent execution exceeds the maximum allowed time."""
    
    def __init__(self, max_execution_time: int, message: str = None):
        self.max_execution_time = max_execution_time
        self.message = message or f"Agent execution exceeded maximum allowed time of {max_execution_time} seconds"
        super().__init__(self.message)
