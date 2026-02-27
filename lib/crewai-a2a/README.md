# crewai-a2a

Agent-to-Agent (A2A) protocol support for CrewAI. Enables agents to discover, authenticate, and communicate with remote A2A-compatible agents.

## Quick Links

[Homepage](https://www.crewai.com/) | [Documentation](https://docs.crewai.com/) | [Community](https://community.crewai.com/)

## Installation

```bash
uv pip install crewai[a2a]
# or
uv add 'crewai[a2a]'
```

## Usage

### Connecting to a Remote A2A Agent

```python
from crewai import Agent
from crewai_a2a import A2AClientConfig

agent = Agent(
    role="Coordinator",
    goal="Delegate research tasks",
    a2a=[
        A2AClientConfig(endpoint="https://research-agent.example.com"),
    ],
)
```

### Exposing an Agent as an A2A Server

```python
from crewai import Agent
from crewai_a2a import A2AServerConfig

agent = Agent(
    role="Researcher",
    goal="Answer research questions",
    a2a_server=A2AServerConfig(
        name="Research Agent",
        description="Answers research questions using web search",
    ),
)
```

## Authentication

### Client Schemes

```python
from crewai_a2a.auth import (
    BearerTokenAuth,
    HTTPBasicAuth,
    APIKeyAuth,
    OAuth2ClientCredentials,
)
from crewai_a2a.config import A2AClientConfig

# Bearer token
A2AClientConfig(
    endpoint="https://agent.example.com",
    auth=BearerTokenAuth(token="my-token"),
)

# API key
A2AClientConfig(
    endpoint="https://agent.example.com",
    auth=APIKeyAuth(api_key="key", location="header", name="X-API-Key"),
)

# OAuth2 client credentials
A2AClientConfig(
    endpoint="https://agent.example.com",
    auth=OAuth2ClientCredentials(
        token_url="https://auth.example.com/token",
        client_id="id",
        client_secret="secret",
    ),
)
```

### Server Schemes

```python
from crewai_a2a.auth import SimpleTokenAuth, OIDCAuth
from crewai_a2a.config import A2AServerConfig

# Simple token validation
A2AServerConfig(auth=SimpleTokenAuth(token="expected-token"))

# OpenID Connect
A2AServerConfig(
    auth=OIDCAuth(
        issuer="https://auth.example.com",
        audience="my-agent",
    ),
)
```

## Update Mechanisms

Control how the client receives task updates from remote agents.

```python
from crewai_a2a.updates import PollingConfig, StreamingConfig, PushNotificationConfig
from crewai_a2a.config import A2AClientConfig


# Polling
A2AClientConfig(
    endpoint="https://agent.example.com",
    updates=PollingConfig(interval=2.0, timeout=60),
)

# Server-Sent Events streaming
A2AClientConfig(
    endpoint="https://agent.example.com",
    updates=StreamingConfig(),
)

# Webhook push notifications
A2AClientConfig(
    endpoint="https://agent.example.com",
    updates=PushNotificationConfig(
        url="https://my-server.example.com/webhook",
        timeout=300,
    ),
)
```

## Extensions

### Client Extensions

Client extensions inject tools, augment prompts, and process responses.

```python
from crewai_a2a.extensions import A2AExtension


class MyExtension(A2AExtension):
    def inject_tools(self, agent):
        ...

    def augment_prompt(self, base_prompt, conversation_state):
        return f"{base_prompt}\n\nAdditional context from extension."
```

### Server Extensions

Server extensions add protocol-level capabilities to your A2A server.

```python
from crewai_a2a.extensions import ServerExtension

class MyServerExtension(ServerExtension):
    uri = "urn:my-org:my-extension"
    description = "Custom protocol extension"

    async def on_request(self, context):
        ...

    async def on_response(self, context, result):
        ...
```

## Transport

Three transport protocols are supported: JSON-RPC (default), gRPC, and HTTP+JSON.

```python
from crewai_a2a.config import ClientTransportConfig, GRPCClientConfig
from crewai_a2a.config import A2AClientConfig


A2AClientConfig(
    endpoint="https://agent.example.com",
    transport=ClientTransportConfig(
        preferred="GRPC",
        grpc=GRPCClientConfig(
            max_send_message_length=4 * 1024 * 1024,
        ),
    ),
)
```
