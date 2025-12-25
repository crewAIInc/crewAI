/**
 * Management API client for CrewManager CRUD operations.
 *
 * Provides functions to manage agents, crews, tasks, and tools
 * via the backend REST API.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// =============================================================================
// Types
// =============================================================================

export interface AgentConfig {
  id: string;
  name?: string;
  role: string;
  goal: string;
  backstory: string;
  llm?: string;
  functionCallingLlm?: string;
  toolIds: string[];
  maxIter: number;
  maxRpm?: number;
  verbose: boolean;
  allowDelegation: boolean;
  tags: string[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface TaskConfig {
  id: string;
  name?: string;
  description: string;
  expectedOutput: string;
  agentId?: string;
  agentRole?: string;
  toolIds: string[];
  contextTaskIds: string[];
  asyncExecution: boolean;
  humanInput: boolean;
  actionBased: boolean;
  tags: string[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface CrewConfig {
  id: string;
  name: string;
  description: string;
  agentIds: string[];
  taskIds: string[];
  process: "sequential" | "hierarchical";
  managerAgentId?: string;
  managerLlm?: string;
  verbose: boolean;
  memory: boolean;
  cache: boolean;
  stream: boolean;
  maxRpm?: number;
  tags: string[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface ToolConfig {
  id: string;
  name: string;
  description: string;
  toolType: string;
  className?: string;
  modulePath?: string;
  initKwargs: Record<string, unknown>;
  envVars: Record<string, string>;
  tags: string[];
  metadata: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
}

export interface ListResult<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
  hasMore: boolean;
}

export interface OperationResult<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp?: string;
}

export interface ExecutionResult {
  success: boolean;
  crewId: string;
  rawOutput: string;
  taskOutputs: TaskOutputSummary[];
  tokenUsage: Record<string, number>;
  executionTimeSeconds: number;
  error?: string;
}

export interface TaskOutputSummary {
  taskId: string;
  taskName?: string;
  agentRole?: string;
  raw: string;
  success: boolean;
}

export interface BuiltinTool {
  name: string;
  description: string;
  module: string;
}

// =============================================================================
// Create Request Types
// =============================================================================

export interface CreateAgentRequest {
  role: string;
  goal: string;
  backstory: string;
  name?: string;
  llm?: string;
  toolIds?: string[];
  maxIter?: number;
  maxRpm?: number;
  verbose?: boolean;
  allowDelegation?: boolean;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface UpdateAgentRequest {
  role?: string;
  goal?: string;
  backstory?: string;
  name?: string;
  llm?: string;
  toolIds?: string[];
  maxIter?: number;
  maxRpm?: number;
  verbose?: boolean;
  allowDelegation?: boolean;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface CreateTaskRequest {
  description: string;
  expectedOutput?: string;
  name?: string;
  agentId?: string;
  agentRole?: string;
  toolIds?: string[];
  contextTaskIds?: string[];
  asyncExecution?: boolean;
  humanInput?: boolean;
  actionBased?: boolean;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface UpdateTaskRequest {
  description?: string;
  expectedOutput?: string;
  name?: string;
  agentId?: string;
  agentRole?: string;
  toolIds?: string[];
  contextTaskIds?: string[];
  asyncExecution?: boolean;
  humanInput?: boolean;
  actionBased?: boolean;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface CreateCrewRequest {
  name: string;
  description?: string;
  agentIds?: string[];
  taskIds?: string[];
  process?: "sequential" | "hierarchical";
  managerAgentId?: string;
  managerLlm?: string;
  verbose?: boolean;
  memory?: boolean;
  cache?: boolean;
  stream?: boolean;
  maxRpm?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface UpdateCrewRequest {
  name?: string;
  description?: string;
  agentIds?: string[];
  taskIds?: string[];
  process?: "sequential" | "hierarchical";
  managerAgentId?: string;
  managerLlm?: string;
  verbose?: boolean;
  memory?: boolean;
  cache?: boolean;
  stream?: boolean;
  maxRpm?: number;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface CreateToolRequest {
  name: string;
  description?: string;
  toolType?: string;
  className?: string;
  modulePath?: string;
  initKwargs?: Record<string, unknown>;
  envVars?: Record<string, string>;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

export interface UpdateToolRequest {
  name?: string;
  description?: string;
  toolType?: string;
  className?: string;
  modulePath?: string;
  initKwargs?: Record<string, unknown>;
  envVars?: Record<string, string>;
  tags?: string[];
  metadata?: Record<string, unknown>;
}

// =============================================================================
// API Helper
// =============================================================================

async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Convert snake_case to camelCase for response data
function toCamelCase(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const camelKey = key.replace(/_([a-z])/g, (_, letter) => letter.toUpperCase());
    if (value && typeof value === "object" && !Array.isArray(value)) {
      result[camelKey] = toCamelCase(value as Record<string, unknown>);
    } else if (Array.isArray(value)) {
      result[camelKey] = value.map((item) =>
        item && typeof item === "object" ? toCamelCase(item as Record<string, unknown>) : item
      );
    } else {
      result[camelKey] = value;
    }
  }
  return result;
}

// Convert camelCase to snake_case for request data
function toSnakeCase(obj: Record<string, unknown>): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    const snakeKey = key.replace(/([A-Z])/g, "_$1").toLowerCase();
    if (value && typeof value === "object" && !Array.isArray(value)) {
      result[snakeKey] = toSnakeCase(value as Record<string, unknown>);
    } else if (Array.isArray(value)) {
      result[snakeKey] = value.map((item) =>
        item && typeof item === "object" ? toSnakeCase(item as Record<string, unknown>) : item
      );
    } else {
      result[snakeKey] = value;
    }
  }
  return result;
}

// =============================================================================
// Agent API
// =============================================================================

export const agentApi = {
  async list(offset = 0, limit = 50, tags?: string[]): Promise<ListResult<AgentConfig>> {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (tags?.length) params.set("tags", tags.join(","));
    const result = await apiRequest<ListResult<AgentConfig>>(`/api/management/agents?${params}`);
    return {
      ...result,
      items: result.items.map((item) => toCamelCase(item as unknown as Record<string, unknown>) as unknown as AgentConfig),
    };
  },

  async get(agentId: string): Promise<AgentConfig> {
    const result = await apiRequest<{ data: AgentConfig }>(`/api/management/agents/${agentId}`);
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as AgentConfig;
  },

  async create(data: CreateAgentRequest): Promise<AgentConfig> {
    const result = await apiRequest<{ data: AgentConfig }>("/api/management/agents", {
      method: "POST",
      body: JSON.stringify(toSnakeCase(data as unknown as Record<string, unknown>)),
    });
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as AgentConfig;
  },

  async update(agentId: string, data: UpdateAgentRequest): Promise<AgentConfig> {
    const result = await apiRequest<{ data: AgentConfig }>(`/api/management/agents/${agentId}`, {
      method: "PUT",
      body: JSON.stringify(toSnakeCase(data as unknown as Record<string, unknown>)),
    });
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as AgentConfig;
  },

  async delete(agentId: string): Promise<void> {
    await apiRequest(`/api/management/agents/${agentId}`, { method: "DELETE" });
  },
};

// =============================================================================
// Task API
// =============================================================================

export const taskApi = {
  async list(offset = 0, limit = 50, tags?: string[]): Promise<ListResult<TaskConfig>> {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (tags?.length) params.set("tags", tags.join(","));
    const result = await apiRequest<ListResult<TaskConfig>>(`/api/management/tasks?${params}`);
    return {
      ...result,
      items: result.items.map((item) => toCamelCase(item as unknown as Record<string, unknown>) as unknown as TaskConfig),
    };
  },

  async get(taskId: string): Promise<TaskConfig> {
    const result = await apiRequest<{ data: TaskConfig }>(`/api/management/tasks/${taskId}`);
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as TaskConfig;
  },

  async create(data: CreateTaskRequest): Promise<TaskConfig> {
    const result = await apiRequest<{ data: TaskConfig }>("/api/management/tasks", {
      method: "POST",
      body: JSON.stringify(toSnakeCase(data as unknown as Record<string, unknown>)),
    });
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as TaskConfig;
  },

  async update(taskId: string, data: UpdateTaskRequest): Promise<TaskConfig> {
    const result = await apiRequest<{ data: TaskConfig }>(`/api/management/tasks/${taskId}`, {
      method: "PUT",
      body: JSON.stringify(toSnakeCase(data as unknown as Record<string, unknown>)),
    });
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as TaskConfig;
  },

  async delete(taskId: string): Promise<void> {
    await apiRequest(`/api/management/tasks/${taskId}`, { method: "DELETE" });
  },
};

// =============================================================================
// Crew API
// =============================================================================

export const crewApi = {
  async list(offset = 0, limit = 50, tags?: string[]): Promise<ListResult<CrewConfig>> {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (tags?.length) params.set("tags", tags.join(","));
    const result = await apiRequest<ListResult<CrewConfig>>(`/api/management/crews?${params}`);
    return {
      ...result,
      items: result.items.map((item) => toCamelCase(item as unknown as Record<string, unknown>) as unknown as CrewConfig),
    };
  },

  async get(crewId: string): Promise<CrewConfig> {
    const result = await apiRequest<{ data: CrewConfig }>(`/api/management/crews/${crewId}`);
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as CrewConfig;
  },

  async create(data: CreateCrewRequest): Promise<CrewConfig> {
    const result = await apiRequest<{ data: CrewConfig }>("/api/management/crews", {
      method: "POST",
      body: JSON.stringify(toSnakeCase(data as unknown as Record<string, unknown>)),
    });
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as CrewConfig;
  },

  async update(crewId: string, data: UpdateCrewRequest): Promise<CrewConfig> {
    const result = await apiRequest<{ data: CrewConfig }>(`/api/management/crews/${crewId}`, {
      method: "PUT",
      body: JSON.stringify(toSnakeCase(data as unknown as Record<string, unknown>)),
    });
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as CrewConfig;
  },

  async delete(crewId: string): Promise<void> {
    await apiRequest(`/api/management/crews/${crewId}`, { method: "DELETE" });
  },

  async execute(crewId: string, inputs: Record<string, unknown> = {}): Promise<ExecutionResult> {
    const result = await apiRequest<ExecutionResult>(`/api/management/crews/${crewId}/execute`, {
      method: "POST",
      body: JSON.stringify({ inputs }),
    });
    return toCamelCase(result as unknown as Record<string, unknown>) as unknown as ExecutionResult;
  },
};

// =============================================================================
// Tool API
// =============================================================================

export const toolApi = {
  async list(offset = 0, limit = 50, tags?: string[]): Promise<ListResult<ToolConfig>> {
    const params = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (tags?.length) params.set("tags", tags.join(","));
    const result = await apiRequest<ListResult<ToolConfig>>(`/api/management/tools?${params}`);
    return {
      ...result,
      items: result.items.map((item) => toCamelCase(item as unknown as Record<string, unknown>) as unknown as ToolConfig),
    };
  },

  async get(toolId: string): Promise<ToolConfig> {
    const result = await apiRequest<{ data: ToolConfig }>(`/api/management/tools/${toolId}`);
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as ToolConfig;
  },

  async create(data: CreateToolRequest): Promise<ToolConfig> {
    const result = await apiRequest<{ data: ToolConfig }>("/api/management/tools", {
      method: "POST",
      body: JSON.stringify(toSnakeCase(data as unknown as Record<string, unknown>)),
    });
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as ToolConfig;
  },

  async update(toolId: string, data: UpdateToolRequest): Promise<ToolConfig> {
    const result = await apiRequest<{ data: ToolConfig }>(`/api/management/tools/${toolId}`, {
      method: "PUT",
      body: JSON.stringify(toSnakeCase(data as unknown as Record<string, unknown>)),
    });
    return toCamelCase(result.data as unknown as Record<string, unknown>) as unknown as ToolConfig;
  },

  async delete(toolId: string): Promise<void> {
    await apiRequest(`/api/management/tools/${toolId}`, { method: "DELETE" });
  },

  async listBuiltin(): Promise<BuiltinTool[]> {
    const result = await apiRequest<{ tools: BuiltinTool[] }>("/api/management/builtin-tools");
    return result.tools;
  },
};

// =============================================================================
// Utility API
// =============================================================================

export const utilityApi = {
  async exportData(format: "json" | "yaml" = "json"): Promise<string> {
    const result = await apiRequest<{ path: string }>(`/api/management/export?format=${format}`, {
      method: "POST",
    });
    return result.path;
  },

  async importData(filePath: string): Promise<void> {
    await apiRequest(`/api/management/import?file_path=${encodeURIComponent(filePath)}`, {
      method: "POST",
    });
  },

  async clearAll(): Promise<void> {
    await apiRequest("/api/management/clear", { method: "POST" });
  },
};
