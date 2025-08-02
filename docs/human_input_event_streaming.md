# Human Input Event Streaming

CrewAI supports real-time event streaming for human input events, allowing clients to receive notifications when human input is required during crew execution. This feature provides an alternative to webhook-only approaches and supports multiple streaming protocols.

## Overview

When a task requires human input (`task.human_input=True`), CrewAI emits events that can be consumed via:

- **WebSocket**: Real-time bidirectional communication
- **Server-Sent Events (SSE)**: Unidirectional server-to-client streaming
- **Long Polling**: HTTP-based polling for events

## Event Types

### HumanInputRequiredEvent

Emitted when human input is required during task execution.

```json
{
  "type": "human_input_required",
  "execution_id": "uuid",
  "crew_id": "uuid", 
  "task_id": "uuid",
  "agent_id": "uuid",
  "prompt": "string",
  "context": "string",
  "timestamp": "ISO8601",
  "event_id": "uuid",
  "reason_flags": {
    "ambiguity": true,
    "missing_field": false
  }
}
```

### HumanInputCompletedEvent

Emitted when human input is completed.

```json
{
  "type": "human_input_completed",
  "execution_id": "uuid",
  "crew_id": "uuid",
  "task_id": "uuid", 
  "agent_id": "uuid",
  "event_id": "uuid",
  "human_feedback": "string",
  "timestamp": "ISO8601"
}
```

## Server Setup

### Installation

Install the server dependencies:

```bash
pip install crewai[server]
```

### Starting the Server

```python
from crewai.server.human_input_server import HumanInputServer

# Start server with authentication
server = HumanInputServer(
    host="localhost",
    port=8000,
    api_key="your-api-key"
)

# Start synchronously
server.start()

# Or start asynchronously
await server.start_async()
```

### Configuration Options

- `host`: Server host (default: "localhost")
- `port`: Server port (default: 8000)
- `api_key`: Optional API key for authentication

## Client Integration

### WebSocket Client

```python
import asyncio
import json
import websockets

async def websocket_client(execution_id: str, api_key: str = None):
    uri = f"ws://localhost:8000/ws/human-input/{execution_id}"
    if api_key:
        uri += f"?token={api_key}"
    
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            event = json.loads(message)
            
            if event['type'] == 'human_input_required':
                print(f"Human input needed: {event['prompt']}")
                print(f"Context: {event['context']}")
            elif event['type'] == 'human_input_completed':
                print(f"Input completed: {event['human_feedback']}")

# Usage
asyncio.run(websocket_client("execution-id", "api-key"))
```

### Server-Sent Events (SSE) Client

```python
import httpx
import json

async def sse_client(execution_id: str, api_key: str = None):
    url = f"http://localhost:8000/events/human-input/{execution_id}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", url, headers=headers) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event.get('type') != 'heartbeat':
                        print(f"Received: {event}")

# Usage
asyncio.run(sse_client("execution-id", "api-key"))
```

### Long Polling Client

```python
import httpx
import asyncio

async def polling_client(execution_id: str, api_key: str = None):
    url = f"http://localhost:8000/poll/human-input/{execution_id}"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    last_event_id = None
    
    async with httpx.AsyncClient() as client:
        while True:
            params = {}
            if last_event_id:
                params["last_event_id"] = last_event_id
            
            response = await client.get(url, headers=headers, params=params)
            data = response.json()
            
            for event in data.get("events", []):
                print(f"Received: {event}")
                last_event_id = event.get('event_id')
            
            await asyncio.sleep(2)  # Poll every 2 seconds

# Usage
asyncio.run(polling_client("execution-id", "api-key"))
```

## API Endpoints

### WebSocket Endpoint

- **URL**: `/ws/human-input/{execution_id}`
- **Protocol**: WebSocket
- **Authentication**: Query parameter `token` (if API key configured)

### SSE Endpoint

- **URL**: `/events/human-input/{execution_id}`
- **Method**: GET
- **Headers**: `Authorization: Bearer <api_key>` (if configured)
- **Response**: `text/event-stream`

### Polling Endpoint

- **URL**: `/poll/human-input/{execution_id}`
- **Method**: GET
- **Headers**: `Authorization: Bearer <api_key>` (if configured)
- **Query Parameters**: 
  - `last_event_id`: Get events after this ID
- **Response**: JSON with `events` array

### Health Check

- **URL**: `/health`
- **Method**: GET
- **Response**: `{"status": "healthy", "timestamp": "..."}`

## Authentication

When an API key is configured, clients must authenticate:

- **WebSocket**: Include `token` query parameter
- **SSE/Polling**: Include `Authorization: Bearer <api_key>` header

## Integration with Crew Execution

The event streaming works automatically with existing crew execution:

```python
from crewai import Agent, Task, Crew

# Create crew with human input task
agent = Agent(...)
task = Task(
    description="...",
    human_input=True,  # This enables human input
    agent=agent
)
crew = Crew(agents=[agent], tasks=[task])

# Start event server (optional)
server = HumanInputServer(port=8000)
server_thread = threading.Thread(target=server.start, daemon=True)
server_thread.start()

# Execute crew - events will be emitted automatically
result = crew.kickoff()
```

## Error Handling

- **Connection Errors**: Clients should implement reconnection logic
- **Authentication Errors**: Server returns 401 for invalid credentials
- **Rate Limiting**: Consider implementing client-side rate limiting for polling

## Best Practices

1. **Use WebSocket** for real-time applications requiring immediate notifications
2. **Use SSE** for one-way streaming with automatic reconnection support
3. **Use Polling** for simple implementations or when WebSocket/SSE aren't available
4. **Implement Authentication** in production environments
5. **Handle Connection Failures** gracefully with retry logic
6. **Filter Events** by execution_id to avoid processing irrelevant events

## Backward Compatibility

This feature is fully backward compatible:

- Existing webhook functionality remains unchanged
- Console-based human input continues to work
- No breaking changes to existing APIs

## Example Applications

- **Web Dashboards**: Real-time crew execution monitoring
- **Mobile Apps**: Push notifications for human input requests  
- **Integration Platforms**: Event-driven workflow automation
- **Monitoring Systems**: Real-time alerting and logging
