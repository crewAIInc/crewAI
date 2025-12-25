import { create } from "zustand";
import type { StreamChunk, ToolCallInfo } from "@/lib/types/stream";

export type Desk = "staff" | "spot" | "futures";

export interface ChatMessage {
  id: string;
  agentId: string;
  agentRole: string;
  taskId: string;
  taskName: string;
  content: string;
  timestamp: string;
  isStreaming: boolean;
  toolCalls: ToolCallInfo[];
  desk: Desk;
}

interface ChatState {
  messages: ChatMessage[];
  streamingMessageId: string | null;
  typingAgentIds: Set<string>;

  // Actions
  addMessage: (chunk: StreamChunk, desk: Desk) => string;
  appendToMessage: (messageId: string, content: string) => void;
  finishStreaming: (messageId: string) => void;
  addToolCall: (messageId: string, toolCall: ToolCallInfo) => void;
  setAgentTyping: (agentId: string, typing: boolean) => void;
  clearMessages: () => void;
  getMessagesByTask: (taskId: string) => ChatMessage[];
  getMessagesByAgent: (agentId: string) => ChatMessage[];
}

let messageCounter = 0;

export const useChatStore = create<ChatState>()((set, get) => ({
  messages: [],
  streamingMessageId: null,
  typingAgentIds: new Set(),

  addMessage: (chunk, desk) => {
    const messageId = `msg-${Date.now()}-${++messageCounter}`;

    set((state) => {
      // Use agent role from chunk, fallback to "Agent" if empty
      const agentRole = chunk.agentRole?.trim() || "Agent";

      const message: ChatMessage = {
        id: messageId,
        agentId: chunk.agentId || "unknown",
        agentRole,
        taskId: chunk.taskId,
        taskName: chunk.taskName,
        content: chunk.content,
        timestamp: chunk.timestamp,
        isStreaming: chunk.chunkType === "text",
        toolCalls: chunk.toolCall ? [chunk.toolCall] : [],
        desk,
      };

      const typingAgentIds = new Set(state.typingAgentIds);
      if (chunk.chunkType === "text") {
        typingAgentIds.add(chunk.agentId);
      }

      return {
        messages: [...state.messages, message],
        streamingMessageId: chunk.chunkType === "text" ? messageId : state.streamingMessageId,
        typingAgentIds,
      };
    });

    return messageId;
  },

  appendToMessage: (messageId, content) => {
    set((state) => {
      const messages = state.messages.map((m) =>
        m.id === messageId ? { ...m, content: m.content + content } : m
      );
      return { messages };
    });
  },

  finishStreaming: (messageId) => {
    set((state) => {
      const messages = state.messages.map((m) =>
        m.id === messageId ? { ...m, isStreaming: false } : m
      );

      const message = state.messages.find((m) => m.id === messageId);
      const typingAgentIds = new Set(state.typingAgentIds);
      if (message) {
        typingAgentIds.delete(message.agentId);
      }

      return {
        messages,
        streamingMessageId: state.streamingMessageId === messageId ? null : state.streamingMessageId,
        typingAgentIds,
      };
    });
  },

  addToolCall: (messageId, toolCall) => {
    set((state) => {
      const messages = state.messages.map((m) =>
        m.id === messageId ? { ...m, toolCalls: [...m.toolCalls, toolCall] } : m
      );
      return { messages };
    });
  },

  setAgentTyping: (agentId, typing) => {
    set((state) => {
      const typingAgentIds = new Set(state.typingAgentIds);
      if (typing) {
        typingAgentIds.add(agentId);
      } else {
        typingAgentIds.delete(agentId);
      }
      return { typingAgentIds };
    });
  },

  clearMessages: () => {
    set({
      messages: [],
      streamingMessageId: null,
      typingAgentIds: new Set(),
    });
  },

  getMessagesByTask: (taskId) => {
    return get().messages.filter((m) => m.taskId === taskId);
  },

  getMessagesByAgent: (agentId) => {
    return get().messages.filter((m) => m.agentId === agentId);
  },
}));
