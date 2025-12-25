import { create } from "zustand";

export type ConnectionStatus = "disconnected" | "connecting" | "connected" | "error";

interface ConnectionState {
  status: ConnectionStatus;
  url: string | null;
  error: string | null;
  reconnectAttempts: number;
  maxReconnectAttempts: number;
  lastConnectedAt: string | null;
  lastDisconnectedAt: string | null;

  // Actions
  setConnecting: (url: string) => void;
  setConnected: () => void;
  setDisconnected: () => void;
  setError: (error: string) => void;
  incrementReconnectAttempts: () => void;
  resetReconnectAttempts: () => void;
  canReconnect: () => boolean;
}

export const useConnectionStore = create<ConnectionState>((set, get) => ({
  status: "disconnected",
  url: null,
  error: null,
  reconnectAttempts: 0,
  maxReconnectAttempts: 5,
  lastConnectedAt: null,
  lastDisconnectedAt: null,

  setConnecting: (url) => {
    set({
      status: "connecting",
      url,
      error: null,
    });
  },

  setConnected: () => {
    set({
      status: "connected",
      error: null,
      reconnectAttempts: 0,
      lastConnectedAt: new Date().toISOString(),
    });
  },

  setDisconnected: () => {
    set({
      status: "disconnected",
      lastDisconnectedAt: new Date().toISOString(),
    });
  },

  setError: (error) => {
    set({
      status: "error",
      error,
    });
  },

  incrementReconnectAttempts: () => {
    set((state) => ({
      reconnectAttempts: state.reconnectAttempts + 1,
    }));
  },

  resetReconnectAttempts: () => {
    set({ reconnectAttempts: 0 });
  },

  canReconnect: () => {
    const { reconnectAttempts, maxReconnectAttempts } = get();
    return reconnectAttempts < maxReconnectAttempts;
  },
}));
