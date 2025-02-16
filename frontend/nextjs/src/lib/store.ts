import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import * as api from "./apiClient";
import { ApiError, wsClient } from "./apiClient";
import { Agent, Crew, Task, TaskStatus, WebSocketEvent } from "./types";

// Define sub-states for better organization
interface CrewState {
  crews: Crew[];
  selectedCrew: Crew | null;
  isLoading: boolean;
  error: ApiError | null;
}

interface TaskState {
  tasks: Task[];
  selectedTask: Task | null;
  isLoading: boolean;
  error: ApiError | null;
}

interface AgentState {
  agents: Agent[];
  selectedAgent: Agent | null;
  isLoading: boolean;
  error: ApiError | null;
}

// Main store state
interface StoreState extends CrewState, TaskState, AgentState {
  // Crew actions
  fetchCrews: () => Promise<void>;
  fetchCrew: (id: string) => Promise<void>;
  createCrew: (crew: Parameters<typeof api.createCrew>[0]) => Promise<void>;
  updateCrew: (
    id: string,
    crew: Parameters<typeof api.updateCrew>[1]
  ) => Promise<void>;
  deleteCrew: (id: string) => Promise<void>;
  setSelectedCrew: (crew: Crew | null) => void;

  // Task actions
  fetchTasks: (crewId?: string) => Promise<void>;
  fetchTask: (id: string) => Promise<void>;
  createTask: (task: Parameters<typeof api.createTask>[0]) => Promise<void>;
  updateTask: (
    id: string,
    task: Parameters<typeof api.updateTask>[1]
  ) => Promise<void>;
  deleteTask: (id: string) => Promise<void>;
  setSelectedTask: (task: Task | null) => void;

  // Agent actions
  fetchAgents: () => Promise<void>;
  fetchAgent: (id: string) => Promise<void>;
  createAgent: (agent: Parameters<typeof api.createAgent>[0]) => Promise<void>;
  updateAgent: (
    id: string,
    agent: Parameters<typeof api.updateAgent>[1]
  ) => Promise<void>;
  deleteAgent: (id: string) => Promise<void>;
  setSelectedAgent: (agent: Agent | null) => void;

  // WebSocket event handlers
  handleWebSocketEvent: (event: WebSocketEvent) => void;

  // Error handling
  clearErrors: () => void;
}

// Create the store with middleware
export const useStore = create<StoreState>()(
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        crews: [],
        selectedCrew: null,
        tasks: [],
        selectedTask: null,
        agents: [],
        selectedAgent: null,
        isLoading: false,
        error: null,

        // Crew actions
        fetchCrews: async () => {
          try {
            set({ isLoading: true, error: null });
            const crews = await api.getCrews();
            set({ crews, isLoading: false });
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        fetchCrew: async (id: string) => {
          try {
            set({ isLoading: true, error: null });
            const crew = await api.getCrew(id);
            set({ selectedCrew: crew, isLoading: false });
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        createCrew: async (crew) => {
          try {
            set({ isLoading: true, error: null });
            const newCrew = await api.createCrew(crew);
            set((state) => ({
              crews: [...state.crews, newCrew],
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        updateCrew: async (id, crew) => {
          try {
            set({ isLoading: true, error: null });
            const updatedCrew = await api.updateCrew(id, crew);
            set((state) => ({
              crews: state.crews.map((c) => (c.id === id ? updatedCrew : c)),
              selectedCrew:
                state.selectedCrew?.id === id
                  ? updatedCrew
                  : state.selectedCrew,
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        deleteCrew: async (id) => {
          try {
            set({ isLoading: true, error: null });
            await api.deleteCrew(id);
            set((state) => ({
              crews: state.crews.filter((c) => c.id !== id),
              selectedCrew:
                state.selectedCrew?.id === id ? null : state.selectedCrew,
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        setSelectedCrew: (crew) => set({ selectedCrew: crew }),

        // Task actions
        fetchTasks: async (crewId) => {
          try {
            set({ isLoading: true, error: null });
            const tasks = await api.getTasks(crewId);
            set({ tasks, isLoading: false });
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        fetchTask: async (id) => {
          try {
            set({ isLoading: true, error: null });
            const task = await api.getTask(id);
            set({ selectedTask: task, isLoading: false });
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        createTask: async (task) => {
          try {
            set({ isLoading: true, error: null });
            const newTask = await api.createTask(task);
            set((state) => ({
              tasks: [...state.tasks, newTask],
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        updateTask: async (id, task) => {
          try {
            set({ isLoading: true, error: null });
            const updatedTask = await api.updateTask(id, task);
            set((state) => ({
              tasks: state.tasks.map((t) => (t.id === id ? updatedTask : t)),
              selectedTask:
                state.selectedTask?.id === id
                  ? updatedTask
                  : state.selectedTask,
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        deleteTask: async (id) => {
          try {
            set({ isLoading: true, error: null });
            await api.deleteTask(id);
            set((state) => ({
              tasks: state.tasks.filter((t) => t.id !== id),
              selectedTask:
                state.selectedTask?.id === id ? null : state.selectedTask,
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        setSelectedTask: (task) => set({ selectedTask: task }),

        // Agent actions
        fetchAgents: async () => {
          try {
            set({ isLoading: true, error: null });
            const agents = await api.getAgents();
            set({ agents, isLoading: false });
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        fetchAgent: async (id) => {
          try {
            set({ isLoading: true, error: null });
            const agent = await api.getAgent(id);
            set({ selectedAgent: agent, isLoading: false });
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        createAgent: async (agent) => {
          try {
            set({ isLoading: true, error: null });
            const newAgent = await api.createAgent(agent);
            set((state) => ({
              agents: [...state.agents, newAgent],
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        updateAgent: async (id, agent) => {
          try {
            set({ isLoading: true, error: null });
            const updatedAgent = await api.updateAgent(id, agent);
            set((state) => ({
              agents: state.agents.map((a) => (a.id === id ? updatedAgent : a)),
              selectedAgent:
                state.selectedAgent?.id === id
                  ? updatedAgent
                  : state.selectedAgent,
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        deleteAgent: async (id) => {
          try {
            set({ isLoading: true, error: null });
            await api.deleteAgent(id);
            set((state) => ({
              agents: state.agents.filter((a) => a.id !== id),
              selectedAgent:
                state.selectedAgent?.id === id ? null : state.selectedAgent,
              isLoading: false,
            }));
          } catch (error) {
            set({ error: error as ApiError, isLoading: false });
          }
        },

        setSelectedAgent: (agent) => set({ selectedAgent: agent }),

        // WebSocket event handlers
        handleWebSocketEvent: (event: WebSocketEvent) => {
          const state = get();
          switch (event.event) {
            case "crew_created":
              set({ crews: [...state.crews, event.payload as Crew] });
              break;
            case "crew_updated":
              const updatedCrew = event.payload as Crew;
              set({
                crews: state.crews.map((c) =>
                  c.id === updatedCrew.id ? updatedCrew : c
                ),
                selectedCrew:
                  state.selectedCrew?.id === updatedCrew.id
                    ? updatedCrew
                    : state.selectedCrew,
              });
              break;
            case "crew_deleted":
              const deletedCrewId = (event.payload as { id: string }).id;
              set({
                crews: state.crews.filter((c) => c.id !== deletedCrewId),
                selectedCrew:
                  state.selectedCrew?.id === deletedCrewId
                    ? null
                    : state.selectedCrew,
              });
              break;
            case "task_created":
              set({ tasks: [...state.tasks, event.payload as Task] });
              break;
            case "task_updated":
              const updatedTask = event.payload as Task;
              set({
                tasks: state.tasks.map((t) =>
                  t.id === updatedTask.id ? updatedTask : t
                ),
                selectedTask:
                  state.selectedTask?.id === updatedTask.id
                    ? updatedTask
                    : state.selectedTask,
              });
              break;
            case "task_deleted":
              const deletedTaskId = (event.payload as { id: string }).id;
              set({
                tasks: state.tasks.filter((t) => t.id !== deletedTaskId),
                selectedTask:
                  state.selectedTask?.id === deletedTaskId
                    ? null
                    : state.selectedTask,
              });
              break;
            case "task_status_changed":
              const { id, status } = event.payload as {
                id: string;
                status: TaskStatus;
              };
              set({
                tasks: state.tasks.map((t) =>
                  t.id === id ? { ...t, status } : t
                ),
                selectedTask:
                  state.selectedTask?.id === id
                    ? { ...state.selectedTask, status }
                    : state.selectedTask,
              });
              break;
            case "agent_message":
              // Handle agent messages if needed
              break;
            case "error":
              set({ error: event.payload as ApiError });
              break;
          }
        },

        // Error handling
        clearErrors: () => set({ error: null }),
      }),
      {
        name: "solnai-store", // Name for localStorage
        partialize: (state) => ({
          // Only persist these fields
          crews: state.crews,
          tasks: state.tasks,
          agents: state.agents,
        }),
      }
    )
  )
);

// Initialize WebSocket connection and set up event handlers
wsClient.on("crew_created", (crew: Crew) =>
  useStore
    .getState()
    .handleWebSocketEvent({
      event: "crew_created",
      payload: crew,
      timestamp: new Date().toISOString(),
    })
);
wsClient.on("crew_updated", (crew: Crew) =>
  useStore
    .getState()
    .handleWebSocketEvent({
      event: "crew_updated",
      payload: crew,
      timestamp: new Date().toISOString(),
    })
);
wsClient.on("crew_deleted", (id: string) =>
  useStore
    .getState()
    .handleWebSocketEvent({
      event: "crew_deleted",
      payload: { id },
      timestamp: new Date().toISOString(),
    })
);
wsClient.on("task_created", (task: Task) =>
  useStore
    .getState()
    .handleWebSocketEvent({
      event: "task_created",
      payload: task,
      timestamp: new Date().toISOString(),
    })
);
wsClient.on("task_updated", (task: Task) =>
  useStore
    .getState()
    .handleWebSocketEvent({
      event: "task_updated",
      payload: task,
      timestamp: new Date().toISOString(),
    })
);
wsClient.on("task_deleted", (id: string) =>
  useStore
    .getState()
    .handleWebSocketEvent({
      event: "task_deleted",
      payload: { id },
      timestamp: new Date().toISOString(),
    })
);
wsClient.on("task_status_changed", (data: { id: string; status: string }) =>
  useStore
    .getState()
    .handleWebSocketEvent({
      event: "task_status_changed",
      payload: data,
      timestamp: new Date().toISOString(),
    })
);
wsClient.on("error", (error: ApiError) =>
  useStore
    .getState()
    .handleWebSocketEvent({
      event: "error",
      payload: error,
      timestamp: new Date().toISOString(),
    })
);

// Connect WebSocket
wsClient.connect();
