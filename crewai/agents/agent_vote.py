from pydantic.v1 import BaseModel, Field

class AgentVote(BaseModel):
    task: str = Field(description="Task to be executed by the agent")
    agent_vote: str = Field(description="Agent that will execute the task")
