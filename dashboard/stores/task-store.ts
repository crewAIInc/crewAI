import { create } from "zustand";
import type { Task, TaskStatus } from "@/lib/types/task";

interface TaskState {
  tasks: Map<string, Task>;
  pendingTaskIds: string[];
  runningTaskIds: string[];
  completedTaskIds: string[];

  // Actions
  addTask: (task: Task) => void;
  updateTaskStatus: (taskId: string, status: TaskStatus) => void;
  updateTaskProgress: (taskId: string, progress: number) => void;
  completeTask: (taskId: string, result: string, duration: number) => void;
  failTask: (taskId: string, error: string) => void;
  getTask: (taskId: string) => Task | undefined;
  getTasksByStatus: (status: TaskStatus) => Task[];
  clearCompletedTasks: () => void;
}

export const useTaskStore = create<TaskState>()((set, get) => ({
  tasks: new Map(),
  pendingTaskIds: [],
  runningTaskIds: [],
  completedTaskIds: [],

  addTask: (task) => {
    set((state) => {
      const tasks = new Map(state.tasks);
      tasks.set(task.id, task);

      const pendingTaskIds = [...state.pendingTaskIds];
      const runningTaskIds = [...state.runningTaskIds];
      const completedTaskIds = [...state.completedTaskIds];

      if (task.status === "pending") {
        pendingTaskIds.push(task.id);
      } else if (task.status === "running") {
        runningTaskIds.push(task.id);
      } else if (task.status === "completed" || task.status === "failed") {
        completedTaskIds.push(task.id);
      }

      return { tasks, pendingTaskIds, runningTaskIds, completedTaskIds };
    });
  },

  updateTaskStatus: (taskId, status) => {
    set((state) => {
      const tasks = new Map(state.tasks);
      const task = tasks.get(taskId);
      if (!task) return state;

      const oldStatus = task.status;
      const updatedTask = { ...task, status };

      if (status === "running") {
        updatedTask.startedAt = new Date().toISOString();
      } else if (status === "completed" || status === "failed") {
        updatedTask.completedAt = new Date().toISOString();
      }

      tasks.set(taskId, updatedTask);

      let pendingTaskIds = [...state.pendingTaskIds];
      let runningTaskIds = [...state.runningTaskIds];
      let completedTaskIds = [...state.completedTaskIds];

      // Remove from old list
      if (oldStatus === "pending") {
        pendingTaskIds = pendingTaskIds.filter((id) => id !== taskId);
      } else if (oldStatus === "running") {
        runningTaskIds = runningTaskIds.filter((id) => id !== taskId);
      } else if (oldStatus === "completed" || oldStatus === "failed") {
        completedTaskIds = completedTaskIds.filter((id) => id !== taskId);
      }

      // Add to new list
      if (status === "pending") {
        pendingTaskIds.push(taskId);
      } else if (status === "running") {
        runningTaskIds.push(taskId);
      } else if (status === "completed" || status === "failed") {
        completedTaskIds.push(taskId);
      }

      return { tasks, pendingTaskIds, runningTaskIds, completedTaskIds };
    });
  },

  updateTaskProgress: (taskId, progress) => {
    set((state) => {
      const tasks = new Map(state.tasks);
      const task = tasks.get(taskId);
      if (task) {
        tasks.set(taskId, { ...task, progress: Math.min(100, Math.max(0, progress)) });
      }
      return { tasks };
    });
  },

  completeTask: (taskId, result, duration) => {
    set((state) => {
      const tasks = new Map(state.tasks);
      const task = tasks.get(taskId);
      if (!task) return state;

      tasks.set(taskId, {
        ...task,
        status: "completed",
        result,
        duration,
        progress: 100,
        completedAt: new Date().toISOString(),
      });

      const runningTaskIds = state.runningTaskIds.filter((id) => id !== taskId);
      const completedTaskIds = [...state.completedTaskIds, taskId];

      return { tasks, runningTaskIds, completedTaskIds };
    });
  },

  failTask: (taskId, error) => {
    set((state) => {
      const tasks = new Map(state.tasks);
      const task = tasks.get(taskId);
      if (!task) return state;

      tasks.set(taskId, {
        ...task,
        status: "failed",
        error,
        completedAt: new Date().toISOString(),
      });

      const pendingTaskIds = state.pendingTaskIds.filter((id) => id !== taskId);
      const runningTaskIds = state.runningTaskIds.filter((id) => id !== taskId);
      const completedTaskIds = [...state.completedTaskIds, taskId];

      return { tasks, pendingTaskIds, runningTaskIds, completedTaskIds };
    });
  },

  getTask: (taskId) => {
    return get().tasks.get(taskId);
  },

  getTasksByStatus: (status) => {
    const { tasks } = get();
    let ids: string[];

    switch (status) {
      case "pending":
        ids = get().pendingTaskIds;
        break;
      case "running":
        ids = get().runningTaskIds;
        break;
      case "completed":
      case "failed":
        ids = get().completedTaskIds;
        break;
      default:
        ids = [];
    }

    return ids
      .map((id) => tasks.get(id))
      .filter((t): t is Task => t !== undefined && t.status === status);
  },

  clearCompletedTasks: () => {
    set((state) => {
      const tasks = new Map(state.tasks);
      state.completedTaskIds.forEach((id) => tasks.delete(id));
      return { tasks, completedTaskIds: [] };
    });
  },
}));
