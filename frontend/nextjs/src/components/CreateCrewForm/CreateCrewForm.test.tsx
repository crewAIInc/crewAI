/// <reference types="jest" />
import * as api from "@/lib/apiClient"; // Import the apiClient
import { Agent, Crew } from "@/lib/types";
import type { jest } from "@jest/globals";
import "@testing-library/jest-dom";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useRouter } from "next/navigation";
import CreateCrewForm from "./CreateCrewForm";

// Mock the apiClient and useRouter
jest.mock("@/lib/apiClient");
jest.mock("next/navigation", () => ({
  useRouter: jest.fn(),
}));

const mockedApi = api as jest.Mocked<typeof api>;
const mockRouterPush = jest.fn();

(useRouter as jest.Mock).mockReturnValue({
  push: mockRouterPush,
});

describe("CreateCrewForm", () => {
  beforeEach(() => {
    jest.clearAllMocks(); // Clear mocks before each test
    mockRouterPush.mockReset();
    // Mock the createAgent function to resolve immediately
    mockedApi.createAgent.mockImplementation(
      async (agentData: Partial<Agent>) => {
        return Promise.resolve({
          id: "agent-id-" + Math.random().toString(36).substring(7), // Generate a unique ID
          ...agentData,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        } as Agent);
      }
    );

    // Mock createCrew to resolve immediately
    mockedApi.createCrew.mockImplementation(async (crewData: Partial<Crew>) => {
      return Promise.resolve({
        id: "crew-id-" + Math.random().toString(36).substring(7), // Generate a unique ID
        ...crewData,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      } as Crew);
    });
  });

  it("renders the form with initial fields", () => {
    render(<CreateCrewForm />);

    expect(
      screen.getByRole("heading", { name: /Create a New Crew/i })
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/Crew Name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Description/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Add Agent/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Create Crew/i })
    ).toBeInTheDocument();
    // Initial agent form
    expect(
      screen.getAllByRole("combobox", { name: /Agent Type/i })
    ).toHaveLength(1);
  });

  it("adds and removes agent fields dynamically", async () => {
    render(<CreateCrewForm />);

    // Initially, one agent form should be present
    expect(screen.getAllByRole("group", { name: /Agent \d+/ })).toHaveLength(1);

    // Add an agent
    fireEvent.click(screen.getByRole("button", { name: /Add Agent/i }));
    await waitFor(() => {
      expect(screen.getAllByRole("group", { name: /Agent \d+/ })).toHaveLength(
        2
      );
    });

    // Remove an agent
    fireEvent.click(screen.getAllByRole("button", { name: /Remove/i })[0]); // Remove first agent
    await waitFor(() => {
      expect(screen.getAllByRole("group", { name: /Agent \d+/ })).toHaveLength(
        1
      );
    });
  });

  it("shows Soln.ai agent fields by default", () => {
    render(<CreateCrewForm />);
    expect(screen.getByLabelText(/Agent Name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Role/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Goal/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Backstory/i)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: /LLM/i })).toBeInTheDocument();
    // These should *not* be present initially
    expect(screen.queryByLabelText(/Temperature/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/Max Tokens/i)).not.toBeInTheDocument();
  });

  it("switches to AutoGen agent fields when selected", async () => {
    render(<CreateCrewForm />);

    const agentTypeSelect = screen.getByRole("combobox", {
      name: /Agent Type/i,
    });
    fireEvent.change(agentTypeSelect, { target: { value: "autogen" } });

    await waitFor(() => {
      expect(agentTypeSelect).toBeInTheDocument();
    });

    // Check if AutoGen fields are present and CrewAI-specific fields are not
    expect(screen.getByLabelText(/Agent Name/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Role/i)).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: /Model/i })
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/Temperature/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Max Tokens/i)).toBeInTheDocument();
    expect(screen.queryByLabelText(/Goal/i)).not.toBeInTheDocument(); // Goal should be gone
    expect(screen.queryByLabelText(/Backstory/i)).not.toBeInTheDocument(); // Backstory should be gone
    expect(
      screen.queryByRole("combobox", { name: /LLM/i })
    ).not.toBeInTheDocument(); // LLM select should be gone
  });

  it("validates required fields", async () => {
    render(<CreateCrewForm />);

    fireEvent.click(screen.getByRole("button", { name: /Create Crew/i }));

    await waitFor(() => {
      expect(screen.getByText("Crew name is required")).toBeInTheDocument();
      expect(screen.getByText("Agent name is required")).toBeInTheDocument();
      expect(screen.getByText("Agent role is required")).toBeInTheDocument();
      expect(screen.getByText("Agent goal is required")).toBeInTheDocument();
      // No LLM selected by default, so this should also be required.
      expect(screen.getByText("LLM is required")).toBeInTheDocument();
    });
  });

  it("handles successful form submission and redirects", async () => {
    render(<CreateCrewForm />);

    // Fill in required fields
    fireEvent.change(screen.getByLabelText(/Crew Name/i), {
      target: { value: "Test Crew" },
    });
    fireEvent.change(screen.getByLabelText(/Agent Name/i), {
      target: { value: "Test Agent" },
    });
    fireEvent.change(screen.getByLabelText(/Role/i), {
      target: { value: "Test Role" },
    });
    fireEvent.change(screen.getByLabelText(/Goal/i), {
      target: { value: "Test Goal" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: /LLM/i }), {
      target: { value: "gpt-4-turbo-preview" },
    });

    // Submit form
    fireEvent.click(screen.getByRole("button", { name: /Create Crew/i }));

    // Wait for submission to complete, and mocked API call.
    await waitFor(() => {
      expect(mockedApi.createCrew).toHaveBeenCalledTimes(1);
      expect(mockedApi.createAgent).toHaveBeenCalledTimes(1);
      expect(mockRouterPush).toHaveBeenCalledWith("/crews");
      expect(
        screen.getByText("Crew created successfully!")
      ).toBeInTheDocument();
    });
  });

  it("handles API errors during form submission", async () => {
    const errorMessage = "Failed to create crew";
    mockedApi.createCrew.mockRejectedValueOnce(new Error(errorMessage));

    render(<CreateCrewForm />);

    // Fill in the form (at least the required fields)
    fireEvent.change(screen.getByLabelText(/Crew Name/i), {
      target: { value: "Test Crew" },
    });
    fireEvent.change(screen.getByLabelText(/Agent Name/i), {
      target: { value: "Test Agent" },
    });
    fireEvent.change(screen.getByLabelText(/Role/i), {
      target: { value: "Test Role" },
    });
    fireEvent.change(screen.getByLabelText(/Goal/i), {
      target: { value: "Test Goal" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "LLM" }), {
      target: { value: "gpt-4-turbo-preview" },
    });

    // Submit the form
    fireEvent.click(screen.getByRole("button", { name: /Create Crew/i }));

    // Wait for the error message to appear
    await waitFor(() => {
      expect(screen.getByText(errorMessage)).toBeInTheDocument(); // Check for general message
    });
  });

  it("shows loading state during submission", async () => {
    mockedApi.createCrew.mockImplementation(
      () => new Promise((resolve) => setTimeout(resolve, 500))
    );
    render(<CreateCrewForm />);
    const submitButton = screen.getByRole("button", { name: /Create Crew/i });

    // Fill in required fields
    fireEvent.change(screen.getByLabelText(/Crew Name/i), {
      target: { value: "Test Crew" },
    });
    fireEvent.change(screen.getByLabelText(/Agent Name/i), {
      target: { value: "Test Agent" },
    });
    fireEvent.change(screen.getByLabelText(/Role/i), {
      target: { value: "Test Role" },
    });
    fireEvent.change(screen.getByLabelText(/Goal/i), {
      target: { value: "Test Goal" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "LLM" }), {
      target: { value: "gpt-4-turbo-preview" },
    });

    fireEvent.click(submitButton);

    expect(submitButton).toBeDisabled();
    expect(
      screen.getByRole("button", { name: /Creating\.\.\./i })
    ).toBeInTheDocument();

    await waitFor(
      () => {
        expect(submitButton).not.toBeDisabled();
      },
      { timeout: 1000 }
    ); // Wait for the button to no longer be disabled (submission complete)
  });

  it("resets form after successful submission", async () => {
    mockedApi.createCrew.mockResolvedValueOnce({ id: "test-id" });
    render(<CreateCrewForm />);

    // Fill in required fields
    fireEvent.change(screen.getByLabelText(/Crew Name/i), {
      target: { value: "Test Crew" },
    });
    fireEvent.change(screen.getByLabelText(/Agent Name/i), {
      target: { value: "Test Agent" },
    });
    fireEvent.change(screen.getByLabelText(/Role/i), {
      target: { value: "Test Role" },
    });
    fireEvent.change(screen.getByLabelText(/Goal/i), {
      target: { value: "Test Goal" },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "LLM" }), {
      target: { value: "gpt-4-turbo-preview" },
    });

    // Submit form
    fireEvent.click(screen.getByRole("button", { name: /Create Crew/i }));

    // Wait for form to reset
    await waitFor(() => {
      expect(screen.getByLabelText(/Crew Name/i)).toHaveValue("");
      // Check if the first agent's name is reset.  Since we add a default agent,
      // we can check if its name field is empty.
      expect(
        (screen.getAllByLabelText(/Agent Name/i)[0] as HTMLInputElement).value
      ).toBe("");
    });
  });

  it("should show the new LLM options for Solnai agents", () => {
    render(<CreateCrewForm />);
    const llmSelect = screen.getByRole("combobox", { name: /LLM/i });

    expect(screen.getByText("GPT-4 Turbo")).toBeInTheDocument();
    expect(screen.getByText("GPT-3.5 Turbo")).toBeInTheDocument();
    expect(screen.getByText("o3-mini")).toBeInTheDocument();
    expect(screen.getByText("Gemini 2.0 Flash")).toBeInTheDocument();
    expect(screen.getByText("Gemini 2.0 Pro")).toBeInTheDocument();
  });

  it("should show the new model options for AutoGen agents", async () => {
    render(<CreateCrewForm />);
    const agentType = screen.getByRole("combobox", { name: "Agent Type" });
    // Switch to AutoGen agent type
    fireEvent.change(agentType, { target: { value: "autogen" } });

    // Wait for the select to be ready.
    await waitFor(() => {
      expect(agentType).toBeInTheDocument();
    });
    const modelSelect = screen.getByRole("combobox", { name: /Model/i });

    expect(screen.getByText("gpt-3.5-turbo-instruct")).toBeInTheDocument();
    expect(screen.getByText("gpt-4-turbo-preview")).toBeInTheDocument();
    expect(screen.getByText("o3-mini")).toBeInTheDocument();
    expect(screen.getByText("Gemini 2.0 Flash")).toBeInTheDocument();
    expect(screen.getByText("Gemini 2.0 Pro")).toBeInTheDocument();
  });
});
