export type StreamChunkType =
  | "text"
  | "tool_call"
  | "task_started"
  | "task_completed"
  | "agent_started"
  | "agent_completed"
  | "error"
  | "heartbeat"
  | "info";

export interface ToolCallInfo {
  toolId: string;
  toolName: string;
  arguments: string;
  result?: string;
  success?: boolean;
}

export interface TaskInfo {
  taskIndex: number;
  taskName: string;
  taskId: string;
  expectedOutput: string;
  totalTasks: number;
  output?: string;
}

export interface AgentInfo {
  agentRole: string;
  agentId: string;
  agentGoal: string;
}

export interface StreamChunk {
  content: string;
  chunkType: StreamChunkType;
  taskIndex: number;
  taskName: string;
  taskId: string;
  agentRole: string;
  agentId: string;
  toolCall?: ToolCallInfo;
  taskInfo?: TaskInfo;
  agentInfo?: AgentInfo;
  timestamp: string;
}

export interface WebSocketMessage<T = unknown> {
  type: "connection" | "stream_chunk" | "task_update" | "agent_update" | "heartbeat" | "error";
  payload: T;
  timestamp: string;
}

export interface ChatMessage {
  id: string;
  agentId: string;
  agentRole: string;
  taskId: string;
  taskName: string;
  content: string;
  type: "text" | "tool_call" | "system";
  toolCall?: ToolCallInfo;
  timestamp: Date;
}
