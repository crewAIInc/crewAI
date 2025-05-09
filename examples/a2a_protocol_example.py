"""
Example of using the A2A protocol with CrewAI.

This example demonstrates how to:
1. Create an agent with A2A protocol support
2. Start an A2A server for the agent
3. Execute a task via the A2A protocol
"""

import asyncio
import os
import uvicorn
from threading import Thread

from crewai import Agent
from crewai.a2a import A2AServer, InMemoryTaskManager


agent = Agent(
    role="Data Analyst",
    goal="Analyze data and provide insights",
    backstory="I am a data analyst with expertise in finding patterns and insights in data.",
    a2a_enabled=True,
    a2a_url="http://localhost:8000",
)


def start_server():
    """Start the A2A server."""
    task_manager = InMemoryTaskManager()
    
    server = A2AServer(task_manager=task_manager)
    
    uvicorn.run(server.app, host="0.0.0.0", port=8000)


async def execute_task_via_a2a():
    """Execute a task via the A2A protocol."""
    await asyncio.sleep(2)
    
    result = await agent.execute_task_via_a2a(
        task_description="Analyze the following data and provide insights: [1, 2, 3, 4, 5]",
        context="This is a simple example of using the A2A protocol.",
    )
    
    print(f"Task result: {result}")


async def main():
    """Run the example."""
    server_thread = Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    
    await execute_task_via_a2a()


if __name__ == "__main__":
    asyncio.run(main())
