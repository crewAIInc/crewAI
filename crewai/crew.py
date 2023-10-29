"""Crew of agents."""

from pydantic import BaseModel, Field

class Crew(BaseModel):
  description: str = Field(description="Description and of the crew being created.")
  goal: str = Field(description="Objective of the crew being created.")
