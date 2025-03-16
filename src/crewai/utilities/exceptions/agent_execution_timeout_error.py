class AgentExecutionTimeoutError(Exception):
    """Exception raised when an agent execution exceeds the maximum allowed time."""
    
    def __init__(
        self, 
        max_execution_time: int, 
        agent_name: str | None = None, 
        task_description: str | None = None, 
        message: str | None = None
    ):
        self.max_execution_time = max_execution_time
        self.agent_name = agent_name
        self.task_description = task_description
        
        # Generate a detailed error message if not provided
        if not message:
            message = f"Agent execution exceeded maximum allowed time of {max_execution_time} seconds"
            if agent_name:
                message += f" for agent: {agent_name}"
            if task_description:
                message += f" while executing task: {task_description}"
        
        self.message = message
        super().__init__(self.message)
