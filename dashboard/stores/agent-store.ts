import { create } from "zustand";
import type { Agent, AgentStatus, Desk } from "@/lib/types/agent";

interface AgentState {
  agents: Map<string, Agent>;
  activeAgentIds: Set<string>;

  // Actions
  setAgents: (agents: Agent[]) => void;
  updateAgentStatus: (agentId: string, status: AgentStatus) => void;
  setAgentActive: (agentId: string, active: boolean) => void;
  getAgentsByDesk: (desk: Desk) => Agent[];
  getActiveAgents: () => Agent[];
  getAgent: (agentId: string) => Agent | undefined;
}

export const useAgentStore = create<AgentState>()((set, get) => ({
  agents: new Map(),
  activeAgentIds: new Set(),

  setAgents: (agents) => {
    set({ agents: new Map(agents.map((a) => [a.id, a])) });
  },

  updateAgentStatus: (agentId, status) => {
    set((state) => {
      const agents = new Map(state.agents);
      const agent = agents.get(agentId);
      if (agent) {
        agents.set(agentId, { ...agent, status });
      }
      const activeAgentIds = new Set(state.activeAgentIds);
      if (status === "active" || status === "busy") {
        activeAgentIds.add(agentId);
      } else {
        activeAgentIds.delete(agentId);
      }
      return { agents, activeAgentIds };
    });
  },

  setAgentActive: (agentId, active) => {
    set((state) => {
      const activeAgentIds = new Set(state.activeAgentIds);
      if (active) {
        activeAgentIds.add(agentId);
      } else {
        activeAgentIds.delete(agentId);
      }
      const agents = new Map(state.agents);
      const agent = agents.get(agentId);
      if (agent) {
        agents.set(agentId, { ...agent, status: active ? "active" : "idle" });
      }
      return { agents, activeAgentIds };
    });
  },

  getAgentsByDesk: (desk) => {
    return Array.from(get().agents.values()).filter((a) => a.desk === desk);
  },

  getActiveAgents: () => {
    const { agents, activeAgentIds } = get();
    return Array.from(activeAgentIds)
      .map((id) => agents.get(id))
      .filter((a): a is Agent => a !== undefined);
  },

  getAgent: (agentId) => {
    return get().agents.get(agentId);
  },
}));
