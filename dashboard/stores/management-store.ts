/**
 * Zustand store for managing CrewAI configurations.
 *
 * Handles state for agents, crews, tasks, and tools with
 * loading states and CRUD operations.
 */

import { create } from "zustand";
import {
  agentApi,
  crewApi,
  taskApi,
  toolApi,
  type AgentConfig,
  type CrewConfig,
  type TaskConfig,
  type ToolConfig,
  type CreateAgentRequest,
  type UpdateAgentRequest,
  type CreateCrewRequest,
  type UpdateCrewRequest,
  type CreateTaskRequest,
  type UpdateTaskRequest,
  type CreateToolRequest,
  type UpdateToolRequest,
} from "@/lib/api/management";

// =============================================================================
// Types
// =============================================================================

interface ManagementState {
  // Agents
  agents: AgentConfig[];
  agentsTotal: number;
  agentsLoading: boolean;
  agentsError: string | null;
  selectedAgent: AgentConfig | null;

  // Crews
  crews: CrewConfig[];
  crewsTotal: number;
  crewsLoading: boolean;
  crewsError: string | null;
  selectedCrew: CrewConfig | null;

  // Tasks
  tasks: TaskConfig[];
  tasksTotal: number;
  tasksLoading: boolean;
  tasksError: string | null;
  selectedTask: TaskConfig | null;

  // Tools
  tools: ToolConfig[];
  toolsTotal: number;
  toolsLoading: boolean;
  toolsError: string | null;
  selectedTool: ToolConfig | null;

  // Agent Actions
  fetchAgents: (offset?: number, limit?: number, tags?: string[]) => Promise<void>;
  fetchAgent: (agentId: string) => Promise<AgentConfig | null>;
  createAgent: (data: CreateAgentRequest) => Promise<AgentConfig>;
  updateAgent: (agentId: string, data: UpdateAgentRequest) => Promise<AgentConfig>;
  deleteAgent: (agentId: string) => Promise<void>;
  selectAgent: (agent: AgentConfig | null) => void;

  // Crew Actions
  fetchCrews: (offset?: number, limit?: number, tags?: string[]) => Promise<void>;
  fetchCrew: (crewId: string) => Promise<CrewConfig | null>;
  createCrew: (data: CreateCrewRequest) => Promise<CrewConfig>;
  updateCrew: (crewId: string, data: UpdateCrewRequest) => Promise<CrewConfig>;
  deleteCrew: (crewId: string) => Promise<void>;
  selectCrew: (crew: CrewConfig | null) => void;

  // Task Actions
  fetchTasks: (offset?: number, limit?: number, tags?: string[]) => Promise<void>;
  fetchTask: (taskId: string) => Promise<TaskConfig | null>;
  createTask: (data: CreateTaskRequest) => Promise<TaskConfig>;
  updateTask: (taskId: string, data: UpdateTaskRequest) => Promise<TaskConfig>;
  deleteTask: (taskId: string) => Promise<void>;
  selectTask: (task: TaskConfig | null) => void;

  // Tool Actions
  fetchTools: (offset?: number, limit?: number, tags?: string[]) => Promise<void>;
  fetchTool: (toolId: string) => Promise<ToolConfig | null>;
  createTool: (data: CreateToolRequest) => Promise<ToolConfig>;
  updateTool: (toolId: string, data: UpdateToolRequest) => Promise<ToolConfig>;
  deleteTool: (toolId: string) => Promise<void>;
  selectTool: (tool: ToolConfig | null) => void;

  // Utility Actions
  clearErrors: () => void;
}

// =============================================================================
// Store
// =============================================================================

export const useManagementStore = create<ManagementState>()((set, get) => ({
  // Initial state
  agents: [],
  agentsTotal: 0,
  agentsLoading: false,
  agentsError: null,
  selectedAgent: null,

  crews: [],
  crewsTotal: 0,
  crewsLoading: false,
  crewsError: null,
  selectedCrew: null,

  tasks: [],
  tasksTotal: 0,
  tasksLoading: false,
  tasksError: null,
  selectedTask: null,

  tools: [],
  toolsTotal: 0,
  toolsLoading: false,
  toolsError: null,
  selectedTool: null,

  // =============================================================================
  // Agent Actions
  // =============================================================================

  fetchAgents: async (offset = 0, limit = 50, tags) => {
    set({ agentsLoading: true, agentsError: null });
    try {
      const result = await agentApi.list(offset, limit, tags);
      set({
        agents: result.items,
        agentsTotal: result.total,
        agentsLoading: false,
      });
    } catch (error) {
      set({
        agentsError: error instanceof Error ? error.message : "Failed to fetch agents",
        agentsLoading: false,
      });
    }
  },

  fetchAgent: async (agentId) => {
    try {
      const agent = await agentApi.get(agentId);
      set({ selectedAgent: agent });
      return agent;
    } catch (error) {
      set({
        agentsError: error instanceof Error ? error.message : "Failed to fetch agent",
      });
      return null;
    }
  },

  createAgent: async (data) => {
    set({ agentsLoading: true, agentsError: null });
    try {
      const agent = await agentApi.create(data);
      set((state) => ({
        agents: [agent, ...state.agents],
        agentsTotal: state.agentsTotal + 1,
        agentsLoading: false,
      }));
      return agent;
    } catch (error) {
      set({
        agentsError: error instanceof Error ? error.message : "Failed to create agent",
        agentsLoading: false,
      });
      throw error;
    }
  },

  updateAgent: async (agentId, data) => {
    set({ agentsLoading: true, agentsError: null });
    try {
      const agent = await agentApi.update(agentId, data);
      set((state) => ({
        agents: state.agents.map((a) => (a.id === agentId ? agent : a)),
        selectedAgent: state.selectedAgent?.id === agentId ? agent : state.selectedAgent,
        agentsLoading: false,
      }));
      return agent;
    } catch (error) {
      set({
        agentsError: error instanceof Error ? error.message : "Failed to update agent",
        agentsLoading: false,
      });
      throw error;
    }
  },

  deleteAgent: async (agentId) => {
    set({ agentsLoading: true, agentsError: null });
    try {
      await agentApi.delete(agentId);
      set((state) => ({
        agents: state.agents.filter((a) => a.id !== agentId),
        agentsTotal: state.agentsTotal - 1,
        selectedAgent: state.selectedAgent?.id === agentId ? null : state.selectedAgent,
        agentsLoading: false,
      }));
    } catch (error) {
      set({
        agentsError: error instanceof Error ? error.message : "Failed to delete agent",
        agentsLoading: false,
      });
      throw error;
    }
  },

  selectAgent: (agent) => {
    set({ selectedAgent: agent });
  },

  // =============================================================================
  // Crew Actions
  // =============================================================================

  fetchCrews: async (offset = 0, limit = 50, tags) => {
    set({ crewsLoading: true, crewsError: null });
    try {
      const result = await crewApi.list(offset, limit, tags);
      set({
        crews: result.items,
        crewsTotal: result.total,
        crewsLoading: false,
      });
    } catch (error) {
      set({
        crewsError: error instanceof Error ? error.message : "Failed to fetch crews",
        crewsLoading: false,
      });
    }
  },

  fetchCrew: async (crewId) => {
    try {
      const crew = await crewApi.get(crewId);
      set({ selectedCrew: crew });
      return crew;
    } catch (error) {
      set({
        crewsError: error instanceof Error ? error.message : "Failed to fetch crew",
      });
      return null;
    }
  },

  createCrew: async (data) => {
    set({ crewsLoading: true, crewsError: null });
    try {
      const crew = await crewApi.create(data);
      set((state) => ({
        crews: [crew, ...state.crews],
        crewsTotal: state.crewsTotal + 1,
        crewsLoading: false,
      }));
      return crew;
    } catch (error) {
      set({
        crewsError: error instanceof Error ? error.message : "Failed to create crew",
        crewsLoading: false,
      });
      throw error;
    }
  },

  updateCrew: async (crewId, data) => {
    set({ crewsLoading: true, crewsError: null });
    try {
      const crew = await crewApi.update(crewId, data);
      set((state) => ({
        crews: state.crews.map((c) => (c.id === crewId ? crew : c)),
        selectedCrew: state.selectedCrew?.id === crewId ? crew : state.selectedCrew,
        crewsLoading: false,
      }));
      return crew;
    } catch (error) {
      set({
        crewsError: error instanceof Error ? error.message : "Failed to update crew",
        crewsLoading: false,
      });
      throw error;
    }
  },

  deleteCrew: async (crewId) => {
    set({ crewsLoading: true, crewsError: null });
    try {
      await crewApi.delete(crewId);
      set((state) => ({
        crews: state.crews.filter((c) => c.id !== crewId),
        crewsTotal: state.crewsTotal - 1,
        selectedCrew: state.selectedCrew?.id === crewId ? null : state.selectedCrew,
        crewsLoading: false,
      }));
    } catch (error) {
      set({
        crewsError: error instanceof Error ? error.message : "Failed to delete crew",
        crewsLoading: false,
      });
      throw error;
    }
  },

  selectCrew: (crew) => {
    set({ selectedCrew: crew });
  },

  // =============================================================================
  // Task Actions
  // =============================================================================

  fetchTasks: async (offset = 0, limit = 50, tags) => {
    set({ tasksLoading: true, tasksError: null });
    try {
      const result = await taskApi.list(offset, limit, tags);
      set({
        tasks: result.items,
        tasksTotal: result.total,
        tasksLoading: false,
      });
    } catch (error) {
      set({
        tasksError: error instanceof Error ? error.message : "Failed to fetch tasks",
        tasksLoading: false,
      });
    }
  },

  fetchTask: async (taskId) => {
    try {
      const task = await taskApi.get(taskId);
      set({ selectedTask: task });
      return task;
    } catch (error) {
      set({
        tasksError: error instanceof Error ? error.message : "Failed to fetch task",
      });
      return null;
    }
  },

  createTask: async (data) => {
    set({ tasksLoading: true, tasksError: null });
    try {
      const task = await taskApi.create(data);
      set((state) => ({
        tasks: [task, ...state.tasks],
        tasksTotal: state.tasksTotal + 1,
        tasksLoading: false,
      }));
      return task;
    } catch (error) {
      set({
        tasksError: error instanceof Error ? error.message : "Failed to create task",
        tasksLoading: false,
      });
      throw error;
    }
  },

  updateTask: async (taskId, data) => {
    set({ tasksLoading: true, tasksError: null });
    try {
      const task = await taskApi.update(taskId, data);
      set((state) => ({
        tasks: state.tasks.map((t) => (t.id === taskId ? task : t)),
        selectedTask: state.selectedTask?.id === taskId ? task : state.selectedTask,
        tasksLoading: false,
      }));
      return task;
    } catch (error) {
      set({
        tasksError: error instanceof Error ? error.message : "Failed to update task",
        tasksLoading: false,
      });
      throw error;
    }
  },

  deleteTask: async (taskId) => {
    set({ tasksLoading: true, tasksError: null });
    try {
      await taskApi.delete(taskId);
      set((state) => ({
        tasks: state.tasks.filter((t) => t.id !== taskId),
        tasksTotal: state.tasksTotal - 1,
        selectedTask: state.selectedTask?.id === taskId ? null : state.selectedTask,
        tasksLoading: false,
      }));
    } catch (error) {
      set({
        tasksError: error instanceof Error ? error.message : "Failed to delete task",
        tasksLoading: false,
      });
      throw error;
    }
  },

  selectTask: (task) => {
    set({ selectedTask: task });
  },

  // =============================================================================
  // Tool Actions
  // =============================================================================

  fetchTools: async (offset = 0, limit = 50, tags) => {
    set({ toolsLoading: true, toolsError: null });
    try {
      const result = await toolApi.list(offset, limit, tags);
      set({
        tools: result.items,
        toolsTotal: result.total,
        toolsLoading: false,
      });
    } catch (error) {
      set({
        toolsError: error instanceof Error ? error.message : "Failed to fetch tools",
        toolsLoading: false,
      });
    }
  },

  fetchTool: async (toolId) => {
    try {
      const tool = await toolApi.get(toolId);
      set({ selectedTool: tool });
      return tool;
    } catch (error) {
      set({
        toolsError: error instanceof Error ? error.message : "Failed to fetch tool",
      });
      return null;
    }
  },

  createTool: async (data) => {
    set({ toolsLoading: true, toolsError: null });
    try {
      const tool = await toolApi.create(data);
      set((state) => ({
        tools: [tool, ...state.tools],
        toolsTotal: state.toolsTotal + 1,
        toolsLoading: false,
      }));
      return tool;
    } catch (error) {
      set({
        toolsError: error instanceof Error ? error.message : "Failed to create tool",
        toolsLoading: false,
      });
      throw error;
    }
  },

  updateTool: async (toolId, data) => {
    set({ toolsLoading: true, toolsError: null });
    try {
      const tool = await toolApi.update(toolId, data);
      set((state) => ({
        tools: state.tools.map((t) => (t.id === toolId ? tool : t)),
        selectedTool: state.selectedTool?.id === toolId ? tool : state.selectedTool,
        toolsLoading: false,
      }));
      return tool;
    } catch (error) {
      set({
        toolsError: error instanceof Error ? error.message : "Failed to update tool",
        toolsLoading: false,
      });
      throw error;
    }
  },

  deleteTool: async (toolId) => {
    set({ toolsLoading: true, toolsError: null });
    try {
      await toolApi.delete(toolId);
      set((state) => ({
        tools: state.tools.filter((t) => t.id !== toolId),
        toolsTotal: state.toolsTotal - 1,
        selectedTool: state.selectedTool?.id === toolId ? null : state.selectedTool,
        toolsLoading: false,
      }));
    } catch (error) {
      set({
        toolsError: error instanceof Error ? error.message : "Failed to delete tool",
        toolsLoading: false,
      });
      throw error;
    }
  },

  selectTool: (tool) => {
    set({ selectedTool: tool });
  },

  // =============================================================================
  // Utility Actions
  // =============================================================================

  clearErrors: () => {
    set({
      agentsError: null,
      crewsError: null,
      tasksError: null,
      toolsError: null,
    });
  },
}));
