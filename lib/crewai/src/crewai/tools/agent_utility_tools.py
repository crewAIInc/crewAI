import uuid
from datetime import datetime
from langchain.tools import tool

class AgentUtilityTools:
    """
    A collection of utility tools to help Agents with 
    grounding (Time) and traceability (IDs).
    """

    @tool("Generate Unique ID")
    def generate_id(tool_input=None):
        """
        Generates a unique Version 4 UUID. 
        Useful for tagging tasks, creating transaction IDs, or tracking agent actions.
        Returns a string ID (e.g., 'a1b2-c3d4...').
        """
        return str(uuid.uuid4())

    @tool("Get Current Time")
    def get_current_time(tool_input=None):
        """
        Returns the current system date and time.
        Useful for agents that need to schedule events, check deadlines, 
        or know 'what day is it today' to avoid hallucinations.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
