"use client";

import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useCallback,
  type ReactNode,
} from "react";
import { WebSocketClient, type WebSocketEventHandler } from "@/lib/websocket";
import { useConnectionStore, type ConnectionStatus } from "@/stores/connection-store";
import { useChatStore, type Desk } from "@/stores/chat-store";
import { useTaskStore } from "@/stores/task-store";
import { useAgentStore } from "@/stores/agent-store";
import type { StreamChunk } from "@/lib/types/stream";

// Extract desk from WebSocket URL like ws://localhost:8000/ws/spot
function getDeskFromUrl(url: string): Desk {
  const match = url.match(/\/ws\/(staff|spot|futures)/);
  return (match?.[1] as Desk) || "spot";
}

interface WebSocketContextValue {
  connect: (url: string) => void;
  disconnect: () => void;
  send: (data: unknown) => void;
  status: ConnectionStatus;
  isConnected: boolean;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

interface WebSocketProviderProps {
  children: ReactNode;
  autoConnect?: boolean;
  defaultUrl?: string;
}

export function WebSocketProvider({
  children,
  autoConnect = false,
  defaultUrl,
}: WebSocketProviderProps) {
  const clientRef = useRef<WebSocketClient | null>(null);

  const { setConnecting, setConnected, setDisconnected, setError, status } =
    useConnectionStore();

  const { addMessage, appendToMessage, finishStreaming, setAgentTyping } = useChatStore();
  const { addTask, updateTaskStatus, updateTaskProgress, completeTask, failTask } =
    useTaskStore();
  const { updateAgentStatus } = useAgentStore();

  // Track current streaming message ID
  const streamingMessageIdRef = useRef<string | null>(null);
  // Track current desk from WebSocket URL
  const currentDeskRef = useRef<Desk>("spot");

  // Track current agent to detect agent changes
  const currentAgentIdRef = useRef<string>("");

  const handleMessage = useCallback(
    (chunk: StreamChunk) => {
      switch (chunk.chunkType) {
        case "text": {
          // Check if this is a new agent or new task
          const isNewAgent = chunk.agentId && chunk.agentId !== currentAgentIdRef.current;
          const isNewTask = chunk.taskId && !streamingMessageIdRef.current;

          // If agent changed, finish current streaming first
          if (isNewAgent && streamingMessageIdRef.current) {
            finishStreaming(streamingMessageIdRef.current);
            streamingMessageIdRef.current = null;
          }

          // If we have an ongoing stream for this agent/task, append to it
          if (streamingMessageIdRef.current && !isNewAgent) {
            appendToMessage(streamingMessageIdRef.current, chunk.content);
          } else {
            // Start a new message with current desk
            const messageId = addMessage(chunk, currentDeskRef.current);
            streamingMessageIdRef.current = messageId;
            currentAgentIdRef.current = chunk.agentId || "";
          }
          break;
        }

        case "tool_call": {
          // Finish any current streaming
          if (streamingMessageIdRef.current) {
            finishStreaming(streamingMessageIdRef.current);
            streamingMessageIdRef.current = null;
          }
          // Add tool call as a new message with current desk
          addMessage(chunk, currentDeskRef.current);
          break;
        }

        case "task_started": {
          // Finish any current streaming
          if (streamingMessageIdRef.current) {
            finishStreaming(streamingMessageIdRef.current);
            streamingMessageIdRef.current = null;
          }

          addTask({
            id: chunk.taskId,
            name: chunk.taskName,
            description: chunk.content,
            agentId: chunk.agentId,
            agentRole: chunk.agentRole,
            status: "running",
            progress: 0,
            createdAt: chunk.timestamp,
          });

          updateAgentStatus(chunk.agentId, "active");
          setAgentTyping(chunk.agentId, true);
          break;
        }

        case "task_completed": {
          // Finish any current streaming
          if (streamingMessageIdRef.current) {
            finishStreaming(streamingMessageIdRef.current);
            streamingMessageIdRef.current = null;
          }

          const duration = parseFloat(chunk.content) || 0;
          completeTask(chunk.taskId, chunk.content, duration);
          updateAgentStatus(chunk.agentId, "idle");
          setAgentTyping(chunk.agentId, false);
          break;
        }

        case "agent_started": {
          updateAgentStatus(chunk.agentId, "active");
          break;
        }

        case "agent_completed": {
          // Finish any current streaming
          if (streamingMessageIdRef.current) {
            finishStreaming(streamingMessageIdRef.current);
            streamingMessageIdRef.current = null;
          }

          updateAgentStatus(chunk.agentId, "idle");
          setAgentTyping(chunk.agentId, false);
          break;
        }

        case "error": {
          // Finish any current streaming
          if (streamingMessageIdRef.current) {
            finishStreaming(streamingMessageIdRef.current);
            streamingMessageIdRef.current = null;
          }

          if (chunk.taskId) {
            failTask(chunk.taskId, chunk.content);
          }
          updateAgentStatus(chunk.agentId, "idle");
          break;
        }

        case "heartbeat":
        case "info": {
          // Info messages and heartbeats don't create chat messages
          // Just acknowledge connection is alive
          break;
        }
      }
    },
    [
      addMessage,
      appendToMessage,
      finishStreaming,
      addTask,
      completeTask,
      failTask,
      updateAgentStatus,
      setAgentTyping,
    ]
  );

  const connect = useCallback(
    (url: string) => {
      if (clientRef.current) {
        clientRef.current.disconnect();
      }

      // Extract desk from URL for message tagging
      currentDeskRef.current = getDeskFromUrl(url);
      setConnecting(url);

      const handlers: WebSocketEventHandler = {
        onOpen: () => {
          setConnected();
        },
        onClose: () => {
          setDisconnected();
          // Finish any current streaming
          if (streamingMessageIdRef.current) {
            finishStreaming(streamingMessageIdRef.current);
            streamingMessageIdRef.current = null;
          }
        },
        onError: () => {
          setError("Connection failed");
        },
        onMessage: handleMessage,
        onReconnecting: (attempt) => {
          console.log(`Reconnecting... attempt ${attempt}`);
        },
      };

      const client = new WebSocketClient({
        url,
        handlers,
        reconnect: true,
        maxReconnectAttempts: 5,
        reconnectInterval: 2000,
      });

      clientRef.current = client;
      client.connect();
    },
    [setConnecting, setConnected, setDisconnected, setError, handleMessage, finishStreaming]
  );

  const disconnect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.disconnect();
      clientRef.current = null;
    }
    setDisconnected();
  }, [setDisconnected]);

  const send = useCallback((data: unknown) => {
    if (clientRef.current) {
      clientRef.current.send(data);
    }
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect && defaultUrl) {
      connect(defaultUrl);
    }

    return () => {
      if (clientRef.current) {
        clientRef.current.disconnect();
      }
    };
  }, [autoConnect, defaultUrl, connect]);

  const value: WebSocketContextValue = {
    connect,
    disconnect,
    send,
    status,
    isConnected: status === "connected",
  };

  return (
    <WebSocketContext.Provider value={value}>{children}</WebSocketContext.Provider>
  );
}

export function useWebSocket() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error("useWebSocket must be used within a WebSocketProvider");
  }
  return context;
}
