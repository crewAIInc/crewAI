/// <reference types="jest" />
import { useStore } from "@/lib/store";
import { Task } from "@/lib/types";
import type { jest } from "@jest/globals";
import "@testing-library/jest-dom";
import { render, screen, waitFor } from "@testing-library/react";
import TaskList from "./TaskList";

// Mock the Zustand store
jest.mock("@/lib/store", () => ({
  useStore: jest.fn(),
}));

const mockTasks: Task[] = [
  {
    id: "1",
    description: "Task 1 Description",
    status: "open",
    crew_id: "crew-1",
    created_at: "2024-03-15T10:00:00Z",
    updated_at: "2024-03-15T10:00:00Z",
  },
  {
    id: "2",
    description: "Task 2 Description",
    status: "completed",
    crew_id: "crew-2",
    created_at: "2024-03-15T12:00:00Z",
    updated_at: "2024-03-15T12:00:00Z",
  },
];

describe("TaskList Component", () => {
  const mockUseStore = useStore as jest.Mock;
  const mockFetchTasks = jest.fn();

  beforeEach(() => {
    // Reset mock implementation before each test
    mockUseStore.mockReset();
    mockFetchTasks.mockReset();
  });

  it("renders loading state", () => {
    mockUseStore.mockReturnValue({
      tasks: [],
      isLoading: true,
      error: null,
      fetchTasks: mockFetchTasks,
    });

    render(<TaskList />);

    // Check for loading skeleton
    expect(screen.getByRole("status")).toBeInTheDocument();
    expect(mockFetchTasks).toHaveBeenCalledTimes(1);
  });

  it("renders error state", () => {
    const errorMessage = "Failed to fetch tasks";
    mockUseStore.mockReturnValue({
      tasks: [],
      isLoading: false,
      error: errorMessage,
      fetchTasks: mockFetchTasks,
    });

    render(<TaskList />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(`Error: ${errorMessage}`)).toBeInTheDocument();
    expect(mockFetchTasks).toHaveBeenCalledTimes(1);
  });

  it("renders empty state when no tasks are present", () => {
    mockUseStore.mockReturnValue({
      tasks: [],
      isLoading: false,
      error: null,
      fetchTasks: mockFetchTasks,
    });

    render(<TaskList />);

    expect(screen.getByText("No tasks found.")).toBeInTheDocument();
    expect(mockFetchTasks).toHaveBeenCalledTimes(1);
  });

  it("renders tasks in a table", async () => {
    mockUseStore.mockReturnValue({
      tasks: mockTasks,
      isLoading: false,
      error: null,
      fetchTasks: mockFetchTasks,
    });

    render(<TaskList />);

    // Wait for the tasks to be rendered
    await waitFor(() => {
      expect(screen.getByText("Task 1 Description")).toBeInTheDocument();
      expect(screen.getByText("Task 2 Description")).toBeInTheDocument();
    });

    // Check table structure
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: /Description/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: /Status/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: /Crew/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: /Actions/i })
    ).toBeInTheDocument();

    // Check task data
    expect(
      screen.getByRole("cell", { name: "Task 1 Description" })
    ).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "open" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "crew-1" })).toBeInTheDocument();

    expect(
      screen.getByRole("cell", { name: "Task 2 Description" })
    ).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "completed" })).toBeInTheDocument();
    expect(screen.getByRole("cell", { name: "crew-2" })).toBeInTheDocument();

    // Check status badges
    const badges = screen.getAllByRole("status");
    expect(badges).toHaveLength(2);
    expect(badges[0]).toHaveTextContent("open");
    expect(badges[1]).toHaveTextContent("completed");

    // Check navigation links
    const viewLinks = screen.getAllByRole("link", { name: /View/i });
    expect(viewLinks).toHaveLength(2);
    expect(viewLinks[0]).toHaveAttribute("href", "/tasks/1");
    expect(viewLinks[1]).toHaveAttribute("href", "/tasks/2");

    expect(mockFetchTasks).toHaveBeenCalledTimes(1);
  });

  it("calls fetchTasks on mount", () => {
    mockUseStore.mockReturnValue({
      tasks: [],
      isLoading: false,
      error: null,
      fetchTasks: mockFetchTasks,
    });

    render(<TaskList />);
    expect(mockFetchTasks).toHaveBeenCalledTimes(1);
  });
});
