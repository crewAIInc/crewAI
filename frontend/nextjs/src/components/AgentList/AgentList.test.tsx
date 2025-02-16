import { useStore } from "@/lib/store";
import type { Agent } from "@/lib/types";
import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import React from "react";
import AgentList from "./AgentList";

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

describe("AgentList", () => {
  const mockUseStore = useStore as jest.Mock;
  const mockFetchAgents = jest.fn();

  beforeEach(() => {
    mockUseStore.mockReset();
    mockFetchAgents.mockReset();
    mockUseStore.mockReturnValue({
      agents: [],
      isLoading: false,
      error: null,
      fetchAgents: mockFetchAgents,
    });
  });

  it("renders loading state", () => {
    mockUseStore.mockReturnValue({
      agents: [],
      isLoading: true,
      error: null,
      fetchAgents: mockFetchAgents,
    });

    render(<AgentList />);
    const skeletons = screen.getAllByRole("status");
    expect(skeletons.length).toBe(6); // We render 6 skeleton cards
  });

  it("renders error state", () => {
    const errorMessage = "Failed to fetch agents";
    mockUseStore.mockReturnValue({
      agents: [],
      isLoading: false,
      error: new Error(errorMessage),
      fetchAgents: mockFetchAgents,
    });

    render(<AgentList />);
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText(`Error: ${errorMessage}`)).toBeInTheDocument();
  });

  it("renders empty state when no agents are present", () => {
    render(<AgentList />);
    expect(screen.getByText("No Agents Found")).toBeInTheDocument();
    expect(
      screen.getByText(/There are no agents in the system yet/)
    ).toBeInTheDocument();
    expect(screen.getByText("Create a New Crew")).toHaveAttribute(
      "href",
      "/crews/new"
    );
  });

  it("renders agent list with both Soln.ai and AutoGen agents", async () => {
    mockUseStore.mockReturnValue({
      agents: [mockSolnAiAgent, mockAutoGenAgent],
      isLoading: false,
      error: null,
      fetchAgents: mockFetchAgents,
    });

    render(<AgentList />);

    // Verify Soln.ai agent rendering
    expect(screen.getByText(mockSolnAiAgent.name)).toBeInTheDocument();
    expect(screen.getByText(mockSolnAiAgent.role)).toBeInTheDocument();
    expect(screen.getByText("solnai")).toBeInTheDocument();
    expect(screen.getByText(mockSolnAiAgent.llm!)).toBeInTheDocument();

    // Verify AutoGen agent rendering
    expect(screen.getByText(mockAutoGenAgent.name)).toBeInTheDocument();
    expect(screen.getByText(mockAutoGenAgent.role)).toBeInTheDocument();
    expect(screen.getByText("autogen")).toBeInTheDocument();
    expect(
      screen.getByText(mockAutoGenAgent.autogen_config!.llm_config!.model)
    ).toBeInTheDocument();

    // Verify links to agent detail pages
    const links = screen.getAllByText("View Details");
    expect(links[0]).toHaveAttribute("href", `/agents/${mockSolnAiAgent.id}`);
    expect(links[1]).toHaveAttribute("href", `/agents/${mockAutoGenAgent.id}`);
  });

  it("calls fetchAgents on mount", () => {
    render(<AgentList />);
    expect(mockFetchAgents).toHaveBeenCalledTimes(1);
  });
});
