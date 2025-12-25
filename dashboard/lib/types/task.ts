export type TaskStatus = "pending" | "running" | "completed" | "failed";

export interface Task {
  id: string;
  index?: number;
  name: string;
  description: string;
  expectedOutput?: string;
  agentId: string;
  agentRole: string;
  status: TaskStatus;
  progress?: number;
  createdAt?: string;
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  result?: string;
  output?: string;
  outputSummary?: string;
  error?: string;
}

export interface TaskUIState {
  tasks: Map<string, Task>;
  tasksByStatus: {
    pending: string[];
    running: string[];
    completed: string[];
    failed: string[];
  };
  currentTask: string | null;
  totalTasks: number;
  completedTasks: number;
}
