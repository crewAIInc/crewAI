// Core interfaces for the Soln.ai frontend
export type AgentType = "solnai" | "autogen";
export type CrewStatus =
  | "active"
  | "inactive"
  | "running"
  | "completed"
  | "failed";
export type TaskStatus = "open" | "in_progress" | "completed" | "failed";

// AutoGen specific configurations
export interface AutoGenAgentConfig {
  name: string;
  llm_config?: {
    model?: string;
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
    presence_penalty?: number;
    frequency_penalty?: number;
    [key: string]: string | number | boolean | undefined | object; // Allow for additional LLM config options
  };
  tools?: string[]; // Available tools for the agent
  system_message?: string; // Custom system message for the agent
}

// Base agent interface with common fields
interface BaseAgent {
  id: string;
  name: string;
  role: string;
  created_at: string;
  updated_at: string;
}

// Soln.ai specific agent interface
interface SolnAiAgent extends BaseAgent {
  agent_type: "solnai";
  goal?: string;
  backstory?: string;
  llm: string;
  allow_delegation?: boolean;
  verbose?: boolean;
  memory?: string;
}

// AutoGen specific agent interface
interface AutoGenAgent extends BaseAgent {
  agent_type: "autogen";
  autogen_config: {
    llm_config: {
      model: string;
      temperature?: number;
      max_tokens?: number;
    };
  };
}

export type Agent = SolnAiAgent | AutoGenAgent;

// Form input types for creating/updating agents
interface BaseAgentInput {
  name: string;
  role: string;
  key?: string; // Add optional key for form management
}

interface SolnAiAgentInput extends BaseAgentInput {
  agent_type: "solnai";
  goal?: string;
  backstory?: string;
  llm: string;
}

interface AutoGenAgentInput extends BaseAgentInput {
  agent_type: "autogen";
  autogen_config: {
    llm_config: {
      model: string;
      temperature?: number;
      max_tokens?: number;
    };
  };
}

export type AgentInput = SolnAiAgentInput | AutoGenAgentInput;
export type CreateAgentInput = AgentInput;
export type UpdateAgentInput = Partial<AgentInput>;

// Form error types for type safety with react-hook-form
export type AgentFormError = {
  name?: { message?: string };
  role?: { message?: string };
  llm?: { message?: string };
} & (
  | {
      agent_type: "solnai";
      goal?: { message?: string };
      backstory?: { message?: string };
      llm?: { message?: string };
    }
  | {
      agent_type: "autogen";
      autogen_config?: {
        llm_config?: {
          model?: { message?: string };
          temperature?: { message?: string };
          max_tokens?: { message?: string };
        };
      };
    }
);

// Form types for creating/updating entities
export type CreateCrewInput = {
  name: string;
  description?: string;
  agents: string[] | AgentInput[];
};

export type UpdateCrewInput = Partial<CreateCrewInput>;

export interface Crew {
  id: string;
  name: string;
  description?: string;
  status: CrewStatus;
  agents: Agent[];
  tasks: string[]; // Array of task IDs
  created_at: string;
  updated_at: string;
  config?: {
    max_consecutive_auto_reply?: number;
    task_timeout?: number;
    [key: string]: string | number | boolean | undefined; // More specific type for additional config options
  };
}

export interface Task {
  id: string;
  description: string;
  status: TaskStatus;
  crew_id: string; // ID of the assigned crew
  agent_id?: string; // ID of the assigned agent
  autogen_recipient?: string; // For AutoGen inter-agent communication
  expected_output?: string; // Expected task output format/description
  context?: string; // Additional context for the task
  dependencies?: string[]; // IDs of tasks that must be completed first
  result?: string; // Task execution result
  error?: string; // Error message if task failed
  created_at: string;
  updated_at: string;
  started_at?: string;
  completed_at?: string;
}

// WebSocket event types
export type WebSocketEventType =
  | "crew_created"
  | "crew_updated"
  | "crew_deleted"
  | "task_created"
  | "task_updated"
  | "task_deleted"
  | "task_status_changed"
  | "agent_message" // For real-time agent communication updates
  | "error";

export interface WebSocketEvent<T = unknown> {
  event: WebSocketEventType;
  payload: T;
  timestamp: string;
}

// API Error types
export interface ApiErrorResponse {
  message: string;
  code: string;
  details?: Record<string, unknown>; // Using Record type instead of any
}

export type CreateTaskInput = Omit<
  Task,
  | "id"
  | "created_at"
  | "updated_at"
  | "started_at"
  | "completed_at"
  | "result"
  | "error"
>;
export type UpdateTaskInput = Partial<CreateTaskInput>;
