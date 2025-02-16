import { Crew, Task, TaskStatus } from "./types";

export const mockCrews: Crew[] = [
  {
    id: "1",
    name: "Research Team Alpha",
    description: "Focused on market research and competitor analysis",
    status: "active",
    created_at: "2024-02-10T10:00:00Z",
    updated_at: "2024-02-10T10:00:00Z",
    tasks: [],
    agents: [
      {
        id: "agent-1",
        name: "Market Researcher",
        role: "Lead Researcher",
        goal: "Analyze market trends and competitor strategies",
        agent_type: "solnai",
        llm: "gpt-4-turbo-preview",
        created_at: "2024-02-10T10:00:00Z",
        updated_at: "2024-02-10T10:00:00Z",
      },
      {
        id: "agent-2",
        name: "Data Analyst",
        role: "Data Processing",
        agent_type: "autogen",
        autogen_config: {
          llm_config: {
            model: "gpt-4-turbo-preview",
            temperature: 0.5,
            max_tokens: 1500,
          },
        },
        created_at: "2024-02-10T10:00:00Z",
        updated_at: "2024-02-10T10:00:00Z",
      },
    ],
  },
  {
    id: "2",
    name: "Content Creation Crew",
    description: "Generates and optimizes content for various platforms",
    status: "active",
    created_at: "2024-02-10T11:00:00Z",
    updated_at: "2024-02-10T11:00:00Z",
    tasks: [],
    agents: [
      {
        id: "agent-3",
        name: "Content Writer",
        role: "Lead Writer",
        goal: "Create engaging content",
        agent_type: "solnai",
        llm: "gpt-4-turbo-preview",
        created_at: "2024-02-10T11:00:00Z",
        updated_at: "2024-02-10T11:00:00Z",
      },
    ],
  },
];

export const mockTasks: Task[] = [
  {
    id: "1",
    crew_id: "1",
    description: "Analyze Q4 2023 market trends",
    status: "in_progress" as TaskStatus,
    created_at: "2024-02-10T10:30:00Z",
    updated_at: "2024-02-10T10:30:00Z",
    completed_at: undefined,
  },
  {
    id: "2",
    crew_id: "1",
    description: "Compile competitor analysis report",
    status: "pending" as TaskStatus,
    created_at: "2024-02-10T10:45:00Z",
    updated_at: "2024-02-10T10:45:00Z",
    completed_at: undefined,
  },
  {
    id: "3",
    crew_id: "2",
    description: "Create social media content calendar",
    status: "completed" as TaskStatus,
    created_at: "2024-02-10T11:30:00Z",
    updated_at: "2024-02-10T12:30:00Z",
    completed_at: "2024-02-10T12:30:00Z",
  },
];

// Helper function to simulate API delay
export const simulateApiDelay = (ms: number = 500) =>
  new Promise((resolve) => setTimeout(resolve, ms));
