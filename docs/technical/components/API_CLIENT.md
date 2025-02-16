# API Client Documentation

## Overview

The API client provides a type-safe interface for interacting with the crewAI backend. It supports both traditional crewAI agents and AutoGen agents, enabling complex multi-agent workflows.

## Core Features

- Type-safe API calls using TypeScript
- Support for both crewAI and AutoGen agents
- Real-time updates via WebSocket
- Comprehensive error handling
- Request caching and optimization

## Overview

The API Client is the core interface between the frontend and the crewAI backend. It provides a type-safe, consistent way to interact with all backend endpoints and handles WebSocket connections for real-time updates.

## Implementation Details

### Directory Structure
```
lib/
├── api/
│   ├── client.ts        # Main API client implementation
│   ├── types.ts         # TypeScript interfaces
│   └── websocket.ts     # WebSocket client implementation
└── config.ts            # API configuration
```

### Type Definitions (`types.ts`)

```typescript
// lib/api/types.ts

export interface Crew {
  id: number;
  name: string;
  agents: number[];
  status: 'Active' | 'Idle' | 'Completed';
}

export interface Task {
  id: number;
  crew_id: number;
  description: string;
  status: 'Pending' | 'In Progress' | 'Completed';
}

export interface Agent {
  id: number;
  name: string;
  role: string;
}

export interface ApiError {
  code: number;
  message: string;
  details?: Record<string, any>;
}

export interface WebSocketEvents {
  crew_created: (crew: Crew) => void;
  crew_updated: (crew: Crew) => void;
  crew_deleted: (crewId: number) => void;
  task_created: (task: Task) => void;
  task_updated: (task: Task) => void;
  task_deleted: (taskId: number) => void;
  task_status_changed: (data: { taskId: number; newStatus: string }) => void;
  agent_log?: (data: { agentId: number; logMessage: string }) => void;
}
```

### API Client Implementation (`client.ts`)

```typescript
// lib/api/client.ts

import { Crew, Task, Agent, ApiError } from './types';

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

class ApiClient {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    try {
      const response = await fetch(`${BASE_URL}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new ApiError(
          response.status,
          errorData?.message || `HTTP error: ${response.status}`
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof TypeError && error.message === 'Failed to fetch') {
        throw new Error('Network Error: Could not connect to the crewAI backend');
      }
      throw error;
    }
  }

  // Crew endpoints
  async getCrews(): Promise<Crew[]> {
    return this.request<Crew[]>('/crews');
  }

  async getCrew(id: number): Promise<Crew> {
    return this.request<Crew>(`/crews/${id}`);
  }

  async createCrew(data: Omit<Crew, 'id'>): Promise<Crew> {
    return this.request<Crew>('/crews', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateCrew(id: number, data: Partial<Crew>): Promise<Crew> {
    return this.request<Crew>(`/crews/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteCrew(id: number): Promise<void> {
    return this.request(`/crews/${id}`, { method: 'DELETE' });
  }

  // Task endpoints
  async getTasks(): Promise<Task[]> {
    return this.request<Task[]>('/tasks');
  }

  async createTask(data: Omit<Task, 'id'>): Promise<Task> {
    return this.request<Task>('/tasks', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateTask(id: number, data: Partial<Task>): Promise<Task> {
    return this.request<Task>(`/tasks/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    });
  }

  async deleteTask(id: number): Promise<void> {
    return this.request(`/tasks/${id}`, { method: 'DELETE' });
  }

  // Agent endpoints
  async getAgent(id: number): Promise<Agent> {
    return this.request<Agent>(`/agents/${id}`);
  }
}

export const apiClient = new ApiClient();
```

### WebSocket Client Implementation (`websocket.ts`)

```typescript
// lib/api/websocket.ts

import { io, Socket } from 'socket.io-client';
import { WebSocketEvents } from './types';

export class WebSocketClient {
  private socket: Socket | null = null;
  private handlers: Partial<WebSocketEvents> = {};

  constructor(
    private url: string = process.env.NEXT_PUBLIC_WEBSOCKET_URL || 'ws://localhost:3001'
  ) {}

  connect(handlers: Partial<WebSocketEvents>) {
    this.handlers = handlers;
    this.socket = io(this.url);

    // Connection event handlers
    this.socket.on('connect', () => {
      console.log('Connected to WebSocket server');
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from WebSocket server');
      this.attemptReconnect();
    });

    this.socket.on('error', (error) => {
      console.error('WebSocket error:', error);
    });

    // Register event handlers
    Object.entries(this.handlers).forEach(([event, handler]) => {
      if (handler) {
        this.socket.on(event, handler);
      }
    });
  }

  private attemptReconnect() {
    if (this.socket) {
      setTimeout(() => {
        console.log('Attempting to reconnect...');
        this.socket?.connect();
      }, 5000); // Retry every 5 seconds
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }
}

export const websocketClient = new WebSocketClient();
```

## Usage Examples

### Basic API Usage

```typescript
// Example: Fetching crews and handling errors
import { apiClient } from '@/lib/api/client';

try {
  const crews = await apiClient.getCrews();
  console.log('Crews:', crews);
} catch (error) {
  console.error('Error fetching crews:', error);
}
```

### WebSocket Integration

```typescript
// Example: Setting up WebSocket listeners in a React component
import { websocketClient } from '@/lib/api/websocket';
import { useEffect } from 'react';

function CrewList() {
  useEffect(() => {
    websocketClient.connect({
      crew_created: (crew) => {
        console.log('New crew created:', crew);
      },
      crew_updated: (crew) => {
        console.log('Crew updated:', crew);
      },
      crew_deleted: (crewId) => {
        console.log('Crew deleted:', crewId);
      },
    });

    return () => {
      websocketClient.disconnect();
    };
  }, []);

  // Component implementation...
}
```

## Error Handling

The API client includes comprehensive error handling:

1. **Network Errors**: Catches and wraps network-related errors
2. **API Errors**: Properly handles and formats API error responses
3. **Type Safety**: Uses TypeScript to ensure type safety at compile time

## Best Practices

1. **Environment Variables**: Always use environment variables for configuration
2. **Error Handling**: Always handle errors in components using try-catch
3. **WebSocket Lifecycle**: Always disconnect WebSocket connections when components unmount
4. **Type Safety**: Use TypeScript interfaces for all data structures
5. **Reconnection Logic**: Implement automatic reconnection for WebSocket connections
