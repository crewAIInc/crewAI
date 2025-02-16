import { v4 as uuidv4 } from "uuid";
import {
  Agent,
  ApiErrorResponse,
  CreateAgentInput,
  CreateCrewInput,
  CreateTaskInput,
  Crew,
  Task,
  TaskStatus,
  UpdateAgentInput,
  UpdateCrewInput,
  UpdateTaskInput,
  WebSocketEvent,
} from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const WS_BASE_URL =
  process.env.NEXT_PUBLIC_WS_BASE_URL || "ws://localhost:8000";
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000; // 1 second

// Custom error class for API errors
export class ApiError extends Error {
  status: number;
  code: string;
  details?: Record<string, unknown>;

  constructor(
    message: string,
    status: number,
    code: string = "UNKNOWN_ERROR",
    details?: Record<string, unknown>
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

// Helper function to add retry logic
const withRetry = async <T>(
  operation: () => Promise<T>,
  retries: number = MAX_RETRIES,
  delay: number = RETRY_DELAY
): Promise<T> => {
  try {
    return await operation();
  } catch (error) {
    if (retries > 0 && error instanceof ApiError && error.status >= 500) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      return withRetry(operation, retries - 1, delay * 2);
    }
    throw error;
  }
};

// Helper function to handle fetch responses and errors
const handleResponse = async <T>(response: Response): Promise<T> => {
  if (!response.ok) {
    const errorData: ApiErrorResponse = await response.json().catch(() => ({
      message: response.statusText,
      code: "UNKNOWN_ERROR",
    }));
    throw new ApiError(
      errorData.message,
      response.status,
      errorData.code,
      errorData.details
    );
  }
  return response.json();
};

// Base fetch function with common options
const fetchWithOptions = async <T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> => {
  const defaultOptions: RequestInit = {
    headers: {
      "Content-Type": "application/json",
      // Add any auth headers here if needed
    },
    ...options,
  };

  return withRetry(async () => {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, defaultOptions);
    return handleResponse<T>(response);
  });
};

// --- Crew API Functions ---
export const getCrews = () => fetchWithOptions<Crew[]>("/crews");

export const getCrew = (id: string) => fetchWithOptions<Crew>(`/crews/${id}`);

export const createCrew = (crewData: CreateCrewInput) =>
  fetchWithOptions<Crew>("/crews", {
    method: "POST",
    body: JSON.stringify(crewData),
  });

export const updateCrew = (id: string, crewData: UpdateCrewInput) =>
  fetchWithOptions<Crew>(`/crews/${id}`, {
    method: "PUT",
    body: JSON.stringify(crewData),
  });

export const deleteCrew = (id: string) =>
  fetchWithOptions<void>(`/crews/${id}`, {
    method: "DELETE",
  });

// --- Task API Functions ---
export const getTasks = (crewId?: string) =>
  fetchWithOptions<Task[]>(crewId ? `/crews/${crewId}/tasks` : "/tasks");

export const getTask = (id: string) => fetchWithOptions<Task>(`/tasks/${id}`);

export const createTask = (taskData: CreateTaskInput) =>
  fetchWithOptions<Task>("/tasks", {
    method: "POST",
    body: JSON.stringify(taskData),
  });

export const updateTask = (id: string, taskData: UpdateTaskInput) =>
  fetchWithOptions<Task>(`/tasks/${id}`, {
    method: "PUT",
    body: JSON.stringify(taskData),
  });

export const deleteTask = (id: string) =>
  fetchWithOptions<void>(`/tasks/${id}`, {
    method: "DELETE",
  });

export const updateTaskStatus = async (
  id: string,
  status: TaskStatus
): Promise<Task> => {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 500));

  // In a real implementation, you would make an API call here
  // to update the task status on the backend.
  // For now, we'll just return a modified task object.
  const task = await getTask(id);
  if (task) {
    const updatedTask: Task = {
      ...task,
      status: status,
      updated_at: new Date().toISOString(),
      completed_at:
        status === "completed" ? new Date().toISOString() : task.completed_at,
    };
    return updatedTask;
  } else {
    throw new ApiError("Task not found to update.", 404, "TASK_NOT_FOUND");
  }
};

// --- Agent API Functions ---
export const getAgents = () => fetchWithOptions<Agent[]>("/agents");

export const getAgent = (id: string) =>
  fetchWithOptions<Agent>(`/agents/${id}`);

// Temporarily replace the real implementation with a mocked one
export const createAgent = async (
  agentData: CreateAgentInput
): Promise<Agent> => {
  // Simulate API delay
  await new Promise((resolve) => setTimeout(resolve, 500));

  // In a real implementation, you would make an API call here
  // and return the newly created agent object from the backend.
  // For now, we'll just generate a unique ID and return the agent data
  const newAgent: Agent = {
    id: uuidv4(), // Use a UUID for the ID
    ...agentData,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
  };

  if (agentData.agent_type === "autogen") {
    newAgent.autogen_config = agentData.autogen_config;
  } else {
    newAgent.llm = agentData.llm;
  }

  return newAgent;
};

export const updateAgent = (id: string, agentData: UpdateAgentInput) =>
  fetchWithOptions<Agent>(`/agents/${id}`, {
    method: "PUT",
    body: JSON.stringify(agentData),
  });

export const deleteAgent = (id: string) =>
  fetchWithOptions<void>(`/agents/${id}`, {
    method: "DELETE",
  });

// --- WebSocket Integration ---
export class WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private eventHandlers: Map<
    WebSocketEvent["event"],
    Set<(data: unknown) => void>
  > = new Map();

  constructor(
    private url: string = WS_BASE_URL,
    private onOpen?: () => void,
    private onClose?: () => void,
    private onError?: (event: Event) => void
  ) {}

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log("WebSocket connected");
      this.reconnectAttempts = 0;
      this.onOpen?.();
    };

    this.ws.onmessage = (event) => {
      try {
        const data: WebSocketEvent = JSON.parse(event.data);
        this.handleEvent(data);
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
      }
    };

    this.ws.onclose = () => {
      console.log("WebSocket disconnected");
      this.onClose?.();
      this.attemptReconnect();
    };

    this.ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      this.onError?.(error);
    };
  }

  private handleEvent(event: WebSocketEvent): void {
    const handlers = this.eventHandlers.get(event.event);
    if (handlers) {
      handlers.forEach((handler) => handler(event.payload));
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error("Max reconnection attempts reached");
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    setTimeout(() => {
      console.log(
        `Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})`
      );
      this.connect();
    }, delay);
  }

  on<T>(
    event: WebSocketEvent["event"],
    handler: (data: T) => void
  ): () => void {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    // Type assertion here is safe because we know the handler will only be called with the correct type
    this.eventHandlers.get(event)!.add(handler as (data: unknown) => void);

    // Return unsubscribe function
    return () => {
      const handlers = this.eventHandlers.get(event);
      if (handlers) {
        handlers.delete(handler as (data: unknown) => void);
        if (handlers.size === 0) {
          this.eventHandlers.delete(event);
        }
      }
    };
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

// Export a singleton instance
export const wsClient = new WebSocketClient();
