import { useStore } from "@/lib/store";
import type { Agent } from "@/lib/types";
import "@testing-library/jest-dom";
import { render, screen, waitFor } from "@testing-library/react";
import React from "react";
import AgentDetail from "./AgentDetail";

// Mock the Zustand store
jest.mock("@/lib/store", () => ({
  useStore: jest.fn(),
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

const mockSolnAiAgent: Agent = {
  id: "agent-1",
  name: "Test Agent",
  role: "Test Role",
  goal: "Test Goal",
  backstory: "Test Backstory",
  agent_type: "solnai",
  llm: "gpt-4-turbo-preview",
  created_at: "2024-03-15T10:00:00Z",
  updated_at: "2024-03-15T10:00:00Z",
};

const mockAutoGenAgent: Agent = {
  id: "agent-2",
  name: "Test AutoGen Agent",
  role: "Test Role AutoGen",
  agent_type: "autogen",
  autogen_config: {
    llm_config: {
      model: "gpt-3.5-turbo-instruct",
    },
  },
  created_at: "2024-03-15T10:00:00Z",
  updated_at: "2024-03-15T10:00:00Z",
};

describe("AgentDetail", () => {
  const mockUseStore = useStore as jest.Mock;
  const mockFetchAgent = jest.fn();

  beforeEach(() => {
    mockUseStore.mockReset();
    mockFetchAgent.mockReset();
    // Provide a default mock implementation for fetchTask
    mockUseStore.mockReturnValue({
      selectedAgent: null,
      isLoading: false,
      error: null,
      fetchAgent: mockFetchAgent, // Ensure fetchTask is always mocked
    });
  });

  it("renders loading state", () => {
    mockUseStore.mockReturnValue({
      selectedAgent: null,
      isLoading: true,
      error: null,
      fetchAgent: jest.fn(),
    });

    render(<AgentDetail agentId="agent-1" />);
    expect(screen.getAllByRole("status")).toHaveLength(4); // 4 skeleton elements
  });

  it("renders error state", () => {
    const error = "Failed to fetch agent";
    mockUseStore.mockReturnValue({
      selectedAgent: null,
      isLoading: false,
      error,
      fetchAgent: jest.fn(),
    });

    render(<AgentDetail agentId="agent-1" />);

    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(`Error: ${error}`)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /back to agents/i })
    ).toBeInTheDocument();
  });

  it("renders agent not found state", () => {
    mockUseStore.mockReturnValue({
      selectedAgent: null,
      isLoading: false,
      error: null,
      fetchAgent: jest.fn(),
    });

    render(<AgentDetail agentId="agent-1" />);

    expect(screen.getByText("Agent Not Found")).toBeInTheDocument();
    expect(screen.getByText(/could not be found/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /back to agents/i })
    ).toBeInTheDocument();
  });

  it("renders Soln.ai agent details", async () => {
    mockUseStore.mockReturnValue({
      selectedAgent: mockSolnAiAgent,
      isLoading: false,
      error: null,
      fetchAgent: jest.fn(),
    });

    render(<AgentDetail agentId="agent-1" />);

    await waitFor(() => {
      expect(screen.getByText(mockSolnAiAgent.name)).toBeInTheDocument();
      expect(
        screen.getByText(`Agent ID: ${mockSolnAiAgent.id}`)
      ).toBeInTheDocument();
      expect(screen.getByText(mockSolnAiAgent.role)).toBeInTheDocument();
      expect(screen.getByText(mockSolnAiAgent.goal!)).toBeInTheDocument();
      expect(screen.getByText(mockSolnAiAgent.backstory!)).toBeInTheDocument();
      expect(screen.getByText(mockSolnAiAgent.llm!)).toBeInTheDocument();
      expect(screen.getByText(/2024-03-15/)).toBeInTheDocument(); // Basic date check
      expect(
        screen.getByRole("button", { name: /back to agents/i })
      ).toBeInTheDocument();
    });
  });

  it("renders AutoGen agent details", async () => {
    mockUseStore.mockReturnValue({
      selectedAgent: mockAutoGenAgent,
      isLoading: false,
      error: null,
      fetchAgent: jest.fn(),
    });

    render(<AgentDetail agentId="agent-2" />);

    await waitFor(() => {
      expect(screen.getByText(mockAutoGenAgent.name)).toBeInTheDocument();
      expect(
        screen.getByText(`Agent ID: ${mockAutoGenAgent.id}`)
      ).toBeInTheDocument();
      expect(screen.getByText(mockAutoGenAgent.role)).toBeInTheDocument();
      expect(screen.getByText("gpt-3.5-turbo-instruct")).toBeInTheDocument();
      expect(screen.getByText(/2024-03-15/)).toBeInTheDocument(); // Basic date check
      expect(
        screen.getByRole("button", { name: /back to agents/i })
      ).toBeInTheDocument();
    });
  });

  it("calls fetchAgent on mount", () => {
    mockUseStore.mockReturnValue({
      selectedAgent: null,
      isLoading: true,
      error: null,
      fetchAgent: mockFetchAgent,
    });
    render(<AgentDetail agentId="agent-123" />);
    expect(mockFetchAgent).toHaveBeenCalledWith("agent-123");
  });
});
