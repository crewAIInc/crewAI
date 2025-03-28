from typing import List, cast

from crewai_tools.tools.website_search.website_search_tool import WebsiteSearchTool
from pydantic import BaseModel, Field

from crewai.lite_agent import LiteAgent


# Define a structured output format
class MovieReview(BaseModel):
    title: str = Field(description="The title of the movie")
    rating: float = Field(description="Rating out of 10")
    pros: List[str] = Field(description="List of positive aspects")
    cons: List[str] = Field(description="List of negative aspects")


# Create a LiteAgent
critic = LiteAgent(
    role="Movie Critic",
    goal="Provide insightful movie reviews",
    backstory="You are an experienced film critic known for balanced, thoughtful reviews.",
    tools=[WebsiteSearchTool()],
    verbose=True,
    response_format=MovieReview,
)

# Use the agent
query = """
Review the movie 'Inception'. Include:
1. Your rating out of 10
2. Key positive aspects
3. Areas that could be improved
"""

result = critic.kickoff(query)

# Access the structured output
review = cast(MovieReview, result.pydantic)
print(f"\nMovie Review: {review.title}")
print(f"Rating: {review.rating}/10")
print("\nPros:")
for pro in review.pros:
    print(f"- {pro}")
print("\nCons:")
for con in review.cons:
    print(f"- {con}")
