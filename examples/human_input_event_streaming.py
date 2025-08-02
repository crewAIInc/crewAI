"""
Example demonstrating how to use the human input event streaming feature.

This example shows how to:
1. Start the human input event server
2. Connect to WebSocket/SSE/polling endpoints
3. Handle human input events in real-time
4. Integrate with crew execution
"""

import asyncio
import json
import threading
import time
from typing import Optional

try:
    import websockets
    import httpx
    from crewai.server.human_input_server import HumanInputServer
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from crewai import Agent, Task, Crew
from crewai.llm import LLM


def start_event_server(port: int = 8000, api_key: Optional[str] = None):
    """Start the human input event server in a separate thread"""
    if not DEPENDENCIES_AVAILABLE:
        print("Server dependencies not available. Install with: pip install crewai[server]")
        return None
    
    server = HumanInputServer(host="localhost", port=port, api_key=api_key)
    
    def run_server():
        server.start()
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    time.sleep(2)
    print(f"Human input event server started on http://localhost:{port}")
    return server


async def websocket_client_example(execution_id: str, api_key: Optional[str] = None):
    """Example WebSocket client for receiving human input events"""
    if not DEPENDENCIES_AVAILABLE:
        print("WebSocket dependencies not available")
        return
    
    uri = f"ws://localhost:8000/ws/human-input/{execution_id}"
    if api_key:
        uri += f"?token={api_key}"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"Connected to WebSocket for execution {execution_id}")
            
            async for message in websocket:
                event_data = json.loads(message)
                print(f"Received WebSocket event: {event_data['type']}")
                
                if event_data['type'] == 'human_input_required':
                    print(f"Human input required for task: {event_data.get('task_id')}")
                    print(f"Prompt: {event_data.get('prompt')}")
                    print(f"Context: {event_data.get('context')}")
                elif event_data['type'] == 'human_input_completed':
                    print(f"Human input completed: {event_data.get('human_feedback')}")
                    
    except Exception as e:
        print(f"WebSocket error: {e}")


async def sse_client_example(execution_id: str, api_key: Optional[str] = None):
    """Example SSE client for receiving human input events"""
    if not DEPENDENCIES_AVAILABLE:
        print("SSE dependencies not available")
        return
    
    url = f"http://localhost:8000/events/human-input/{execution_id}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", url, headers=headers) as response:
                print(f"Connected to SSE for execution {execution_id}")
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        event_data = json.loads(line[6:])
                        if event_data.get('type') != 'heartbeat':
                            print(f"Received SSE event: {event_data['type']}")
                            
                            if event_data['type'] == 'human_input_required':
                                print(f"Human input required for task: {event_data.get('task_id')}")
                                print(f"Prompt: {event_data.get('prompt')}")
                            elif event_data['type'] == 'human_input_completed':
                                print(f"Human input completed: {event_data.get('human_feedback')}")
                                
    except Exception as e:
        print(f"SSE error: {e}")


async def polling_client_example(execution_id: str, api_key: Optional[str] = None):
    """Example polling client for receiving human input events"""
    if not DEPENDENCIES_AVAILABLE:
        print("Polling dependencies not available")
        return
    
    url = f"http://localhost:8000/poll/human-input/{execution_id}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    last_event_id = None
    
    try:
        async with httpx.AsyncClient() as client:
            print(f"Starting polling for execution {execution_id}")
            
            while True:
                params = {}
                if last_event_id:
                    params["last_event_id"] = last_event_id
                
                response = await client.get(url, headers=headers, params=params)
                data = response.json()
                
                for event in data.get("events", []):
                    print(f"Received polling event: {event['type']}")
                    
                    if event['type'] == 'human_input_required':
                        print(f"Human input required for task: {event.get('task_id')}")
                        print(f"Prompt: {event.get('prompt')}")
                    elif event['type'] == 'human_input_completed':
                        print(f"Human input completed: {event.get('human_feedback')}")
                    
                    last_event_id = event.get('event_id')
                
                await asyncio.sleep(2)
                
    except Exception as e:
        print(f"Polling error: {e}")


def create_sample_crew():
    """Create a sample crew that requires human input"""
    
    llm = LLM(model="gpt-4o-mini")
    
    agent = Agent(
        role="Research Assistant",
        goal="Help with research tasks and get human feedback",
        backstory="You are a helpful research assistant that works with humans to complete tasks.",
        llm=llm,
        verbose=True
    )
    
    task = Task(
        description="Research the latest trends in AI and provide a summary. Ask for human feedback on the findings.",
        expected_output="A comprehensive summary of AI trends with human feedback incorporated.",
        agent=agent,
        human_input=True
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True
    )
    
    return crew


async def main():
    """Main example function"""
    print("CrewAI Human Input Event Streaming Example")
    print("=" * 50)
    
    api_key = "demo-api-key"
    
    server = start_event_server(port=8000, api_key=api_key)
    if not server:
        return
    
    crew = create_sample_crew()
    execution_id = str(crew.id)
    
    print(f"Crew execution ID: {execution_id}")
    print("\nStarting event stream clients...")
    
    websocket_task = asyncio.create_task(
        websocket_client_example(execution_id, api_key)
    )
    
    sse_task = asyncio.create_task(
        sse_client_example(execution_id, api_key)
    )
    
    polling_task = asyncio.create_task(
        polling_client_example(execution_id, api_key)
    )
    
    await asyncio.sleep(1)
    
    print("\nStarting crew execution...")
    print("Note: This will prompt for human input in the console.")
    print("The event streams above will also receive the events in real-time.")
    
    def run_crew():
        try:
            result = crew.kickoff()
            print(f"\nCrew execution completed: {result}")
        except Exception as e:
            print(f"Crew execution error: {e}")
    
    crew_thread = threading.Thread(target=run_crew)
    crew_thread.start()
    
    await asyncio.sleep(30)
    
    websocket_task.cancel()
    sse_task.cancel()
    polling_task.cancel()
    
    crew_thread.join(timeout=5)
    
    print("\nExample completed!")


if __name__ == "__main__":
    if DEPENDENCIES_AVAILABLE:
        asyncio.run(main())
    else:
        print("Dependencies not available. Install with: pip install crewai[server]")
        print("This example requires FastAPI, uvicorn, websockets, and httpx.")
