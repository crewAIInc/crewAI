"""
FastAPI Streaming Integration Example for CrewAI

This example demonstrates how to integrate CrewAI with FastAPI to stream
crew execution events in real-time using Server-Sent Events (SSE).

Installation:
    pip install crewai fastapi uvicorn

Usage:
    python fastapi_streaming_example.py

Then visit:
    http://localhost:8000/docs for the API documentation
    http://localhost:8000/stream?topic=AI to see streaming in action
"""

import json
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from crewai import Agent, Crew, Task

app = FastAPI(title="CrewAI Streaming API")


class ResearchRequest(BaseModel):
    topic: str


def create_research_crew(topic: str) -> Crew:
    """Create a research crew for the given topic."""
    researcher = Agent(
        role="Researcher",
        goal=f"Research and analyze information about {topic}",
        backstory="You're an expert researcher with deep knowledge in various fields.",
        verbose=True,
    )

    task = Task(
        description=f"Research and provide a comprehensive summary about {topic}",
        expected_output="A detailed summary with key insights",
        agent=researcher,
    )

    return Crew(agents=[researcher], tasks=[task], verbose=True)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CrewAI Streaming API",
        "endpoints": {
            "/stream": "GET - Stream crew execution events (query param: topic)",
            "/research": "POST - Execute crew and return final result",
        },
    }


@app.get("/stream")
async def stream_crew_execution(topic: str = "artificial intelligence"):
    """
    Stream crew execution events in real-time using Server-Sent Events.

    Args:
        topic: The research topic (default: "artificial intelligence")

    Returns:
        StreamingResponse with text/event-stream content type
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events from crew execution."""
        crew = create_research_crew(topic)

        try:
            for event in crew.kickoff_stream(inputs={"topic": topic}):
                event_data = json.dumps(event)
                yield f"data: {event_data}\n\n"

            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as e:
            error_event = {"type": "error", "data": {"message": str(e)}}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/research")
async def research_topic(request: ResearchRequest):
    """
    Execute crew research and return the final result.

    Args:
        request: ResearchRequest with topic field

    Returns:
        JSON response with the research result
    """
    crew = create_research_crew(request.topic)

    try:
        result = crew.kickoff(inputs={"topic": request.topic})
        return {
            "success": True,
            "topic": request.topic,
            "result": result.raw,
            "usage_metrics": (
                result.token_usage.model_dump() if result.token_usage else None
            ),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/stream-filtered")
async def stream_filtered_events(
    topic: str = "artificial intelligence", event_types: str = "llm_stream_chunk"
):
    """
    Stream only specific event types.

    Args:
        topic: The research topic
        event_types: Comma-separated list of event types to include

    Returns:
        StreamingResponse with filtered events
    """
    allowed_types = set(event_types.split(","))

    async def event_generator() -> AsyncGenerator[str, None]:
        crew = create_research_crew(topic)

        try:
            for event in crew.kickoff_stream(inputs={"topic": topic}):
                if event["type"] in allowed_types:
                    event_data = json.dumps(event)
                    yield f"data: {event_data}\n\n"

            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as e:
            error_event = {"type": "error", "data": {"message": str(e)}}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


if __name__ == "__main__":
    import uvicorn

    print("Starting CrewAI Streaming API...")
    print("Visit http://localhost:8000/docs for API documentation")
    print("Try: http://localhost:8000/stream?topic=quantum%20computing")

    uvicorn.run(app, host="0.0.0.0", port=8000)
