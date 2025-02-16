/// <reference types="jest" />
import * as api from "@/lib/apiClient";
import { useStore } from "@/lib/store";
import type { Task, TaskStatus } from "@/lib/types";
import "@testing-library/jest-dom/extend-expect";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import React from "react";
import TaskDetail from "./TaskDetail";

// Extend the Jest matchers
declare global {
  namespace jest {
    interface Matchers<R> {
      toBeInTheDocument(): R;
      toHaveTextContent(text: string): R;
    }
  }
}

// Mock the apiClient
jest.mock("@/lib/apiClient", () => ({
  updateTaskStatus: jest.fn(),
}));

// Mock the Zustand store with proper type casting
jest.mock("@/lib/store", () => ({
  useStore: jest.fn().mockImplementation(() => ({
    task: null,
    isLoading: false,
    error: null,
    fetchTask: jest.fn(),
  })),
}));

// Mock next/link
jest.mock("next/link", () => ({
  __esModule: true,
  default: ({
    children,
    href,
  }: {
    children: React.ReactNode;
    href: string;
  }) => <a href={href}>{children}</a>,
}));

const mockTask: Task = {
  id: "task-123",
  description: "Test Task",
  status: "in_progress",
  crew_id: "crew-456",
  created_at: "2024-03-15T10:00:00Z",
  updated_at: "2024-03-15T13:00:00Z",
};

describe("TaskDetail", () => {
  // Cast useStore to unknown first, then to jest.Mock to avoid type errors
  const mockUseStore = useStore as unknown as jest.Mock;
  const mockFetchTask = jest.fn();
  const mockUpdateTaskStatus = api.updateTaskStatus as jest.Mock;

  beforeEach(() => {
    mockUseStore.mockReset();
    mockFetchTask.mockReset();
    mockUpdateTaskStatus.mockReset();
    mockUseStore.mockImplementation(() => ({
      task: null,
      isLoading: false,
      error: null,
      fetchTask: mockFetchTask,
    }));
  });

  it("renders loading state", () => {
    mockUseStore.mockImplementation(() => ({
      task: null,
      isLoading: true,
      error: null,
      fetchTask: jest.fn(),
    }));

    render(<TaskDetail taskId="task-123" />);
    expect(screen.getAllByRole("status")).toHaveLength(4);
  });

  it("renders error state", () => {
    const error = "Failed to fetch task";
    mockUseStore.mockImplementation(() => ({
      task: null,
      isLoading: false,
      error,
      fetchTask: jest.fn(),
    }));

    render(<TaskDetail taskId="task-123" />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(`Error: ${error}`)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /back to tasks/i })
    ).toBeInTheDocument();
  });

  it("renders task not found state", () => {
    mockUseStore.mockImplementation(() => ({
      task: null,
      isLoading: false,
      error: null,
      fetchTask: jest.fn(),
    }));

    render(<TaskDetail taskId="task-123" />);

    expect(screen.getByText("Task Not Found")).toBeInTheDocument();
    expect(screen.getByText(/could not be found/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /back to tasks/i })
    ).toBeInTheDocument();
  });

  it("renders task details", async () => {
    mockUseStore.mockImplementation(() => ({
      task: mockTask,
      isLoading: false,
      error: null,
      fetchTask: jest.fn(),
    }));

    render(<TaskDetail taskId="task-123" />);

    await waitFor(() => {
      expect(screen.getByText(mockTask.description)).toBeInTheDocument();
      expect(screen.getByText(`Task ID: ${mockTask.id}`)).toBeInTheDocument();
      expect(screen.getByText(mockTask.status)).toBeInTheDocument();
      expect(screen.getByText(mockTask.crew_id)).toBeInTheDocument();
      expect(screen.getByText(/2024-03-15/)).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /mark as completed/i })
      ).toBeInTheDocument();
    });
  });

  it("handles task completion successfully", async () => {
    mockUseStore.mockImplementation(() => ({
      task: mockTask,
      isLoading: false,
      error: null,
      fetchTask: mockFetchTask,
    }));

    mockUpdateTaskStatus.mockResolvedValueOnce({
      ...mockTask,
      status: "completed",
      updated_at: new Date().toISOString(),
    });

    render(<TaskDetail taskId="task-123" />);

    const completeButton = screen.getByRole("button", {
      name: /mark as completed/i,
    });
    expect(completeButton).toBeInTheDocument();

    fireEvent.click(completeButton);

    // Button should be disabled and show loading state
    expect(completeButton).toBeDisabled();
    expect(completeButton).toHaveTextContent("Completing...");

    await waitFor(() => {
      expect(mockUpdateTaskStatus).toHaveBeenCalledWith(
        mockTask.id,
        "completed"
      );
      expect(mockFetchTask).toHaveBeenCalledWith(mockTask.id);
    });
  });

  it("handles task completion error", async () => {
    mockUseStore.mockImplementation(() => ({
      task: mockTask,
      isLoading: false,
      error: null,
      fetchTask: mockFetchTask,
    }));

    const errorMessage = "Failed to update task";
    mockUpdateTaskStatus.mockRejectedValueOnce(new Error(errorMessage));

    render(<TaskDetail taskId="task-123" />);

    const completeButton = screen.getByRole("button", {
      name: /mark as completed/i,
    });
    fireEvent.click(completeButton);

    await waitFor(() => {
      const errorElement = screen.queryByText(errorMessage);
      expect(errorElement).not.toBeNull();
      expect(errorElement).toBeInTheDocument();
      expect(completeButton).not.toBeDisabled();
      expect(completeButton).toHaveTextContent("Mark as Completed");
    });
  });

  it("does not show complete button for completed tasks", async () => {
    const completedTask = { ...mockTask, status: "completed" as TaskStatus };
    mockUseStore.mockImplementation(() => ({
      task: completedTask,
      isLoading: false,
      error: null,
      fetchTask: mockFetchTask,
    }));

    render(<TaskDetail taskId="task-123" />);

    await waitFor(() => {
      expect(
        screen.queryByRole("button", { name: /mark as completed/i })
      ).not.toBeInTheDocument();
    });
  });

  it("calls fetchTask on mount", () => {
    const fetchTaskMock = jest.fn();
    mockUseStore.mockImplementation(() => ({
      task: null,
      isLoading: true,
      error: null,
      fetchTask: fetchTaskMock,
    }));

    render(<TaskDetail taskId="task-123" />);
    expect(fetchTaskMock).toHaveBeenCalledWith("task-123");
  });
});
