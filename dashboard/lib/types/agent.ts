export type Desk = "staff" | "spot" | "futures";

export type AgentStatus = "idle" | "active" | "busy";

export type RoleCategory =
  | "leadership"
  | "systematic"
  | "discretionary"
  | "arbitrage"
  | "research"
  | "execution"
  | "event"
  | "market_making"
  | "risk"
  | "operations"
  | "carry"
  | "microstructure"
  | "swing";

export interface Agent {
  id: string;
  number: number; // 0-73 (STAFF 0-9, Spot 1-32, Futures 33-64)
  name: string;
  role: string;
  goal: string;
  backstory: string;
  desk: Desk;
  category: RoleCategory;
  color: string;
  textColor: string;
  status: AgentStatus;
  usesHeavyLLM: boolean;
  allowsDelegation: boolean;
}

export interface AgentUIState {
  agents: Map<string, Agent>;
  activeAgents: Set<string>;
  currentAgent: string | null;
}
