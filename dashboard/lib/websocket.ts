import type { StreamChunk } from "@/lib/types/stream";

export type WebSocketEventHandler = {
  onOpen?: () => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (error: Event) => void;
  onMessage?: (chunk: StreamChunk) => void;
  onReconnecting?: (attempt: number) => void;
};

export interface WebSocketClientOptions {
  url: string;
  reconnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  handlers?: WebSocketEventHandler;
}

export class WebSocketClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnect: boolean;
  private maxReconnectAttempts: number;
  private reconnectInterval: number;
  private reconnectAttempts = 0;
  private reconnectTimeoutId: ReturnType<typeof setTimeout> | null = null;
  private handlers: WebSocketEventHandler;
  private isManualClose = false;

  constructor(options: WebSocketClientOptions) {
    this.url = options.url;
    this.reconnect = options.reconnect ?? true;
    this.maxReconnectAttempts = options.maxReconnectAttempts ?? 5;
    this.reconnectInterval = options.reconnectInterval ?? 3000;
    this.handlers = options.handlers ?? {};
  }

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.isManualClose = false;

    try {
      this.ws = new WebSocket(this.url);
      this.setupEventListeners();
    } catch (error) {
      console.error("[WebSocket] Connection error:", error);
      this.handlers.onError?.(error as Event);
      this.attemptReconnect();
    }
  }

  private setupEventListeners(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log("[WebSocket] Connected to", this.url);
      this.reconnectAttempts = 0;
      this.handlers.onOpen?.();
    };

    this.ws.onclose = (event) => {
      console.log("[WebSocket] Disconnected:", event.code, event.reason);
      this.handlers.onClose?.(event);

      if (!this.isManualClose && this.reconnect) {
        this.attemptReconnect();
      }
    };

    this.ws.onerror = (error) => {
      console.error("[WebSocket] Error:", error);
      this.handlers.onError?.(error);
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const chunk = this.parseStreamChunk(data);
        if (chunk) {
          this.handlers.onMessage?.(chunk);
        }
      } catch (error) {
        console.error("[WebSocket] Failed to parse message:", error);
      }
    };
  }

  private parseStreamChunk(data: unknown): StreamChunk | null {
    if (!data || typeof data !== "object") return null;

    const raw = data as Record<string, unknown>;

    return {
      content: String(raw.content ?? ""),
      chunkType: (raw.chunk_type ?? raw.chunkType ?? "text") as StreamChunk["chunkType"],
      taskIndex: Number(raw.task_index ?? raw.taskIndex ?? 0),
      agentRole: String(raw.agent_role ?? raw.agentRole ?? ""),
      agentId: String(raw.agent_id ?? raw.agentId ?? ""),
      taskName: String(raw.task_name ?? raw.taskName ?? ""),
      taskId: String(raw.task_id ?? raw.taskId ?? ""),
      timestamp: String(raw.timestamp ?? new Date().toISOString()),
      toolCall: raw.tool_call ?? raw.toolCall
        ? {
            toolId: String((raw.tool_call as Record<string, unknown>)?.tool_id ?? (raw.toolCall as Record<string, unknown>)?.toolId ?? ""),
            toolName: String((raw.tool_call as Record<string, unknown>)?.tool_name ?? (raw.tool_call as Record<string, unknown>)?.name ?? (raw.toolCall as Record<string, unknown>)?.toolName ?? ""),
            arguments: String((raw.tool_call as Record<string, unknown>)?.arguments ?? (raw.toolCall as Record<string, unknown>)?.arguments ?? "{}"),
            result: String((raw.tool_call as Record<string, unknown>)?.result ?? (raw.toolCall as Record<string, unknown>)?.result ?? ""),
            success: Boolean((raw.tool_call as Record<string, unknown>)?.success ?? (raw.toolCall as Record<string, unknown>)?.success ?? true),
          }
        : undefined,
    };
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log("[WebSocket] Max reconnect attempts reached");
      return;
    }

    this.reconnectAttempts++;
    console.log(
      `[WebSocket] Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`
    );

    this.handlers.onReconnecting?.(this.reconnectAttempts);

    this.reconnectTimeoutId = setTimeout(() => {
      this.connect();
    }, this.reconnectInterval * Math.min(this.reconnectAttempts, 3));
  }

  disconnect(): void {
    this.isManualClose = true;

    if (this.reconnectTimeoutId) {
      clearTimeout(this.reconnectTimeoutId);
      this.reconnectTimeoutId = null;
    }

    if (this.ws) {
      this.ws.close(1000, "Client disconnect");
      this.ws = null;
    }
  }

  send(data: unknown): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn("[WebSocket] Cannot send, not connected");
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  get connectionState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED;
  }

  updateHandlers(handlers: Partial<WebSocketEventHandler>): void {
    this.handlers = { ...this.handlers, ...handlers };
  }
}

// Singleton factory for crew connections
const crewConnections = new Map<string, WebSocketClient>();

export function getCrewConnection(crewId: string, baseUrl: string): WebSocketClient {
  const key = `${baseUrl}/${crewId}`;

  if (!crewConnections.has(key)) {
    const client = new WebSocketClient({
      url: `${baseUrl}/ws/${crewId}`,
      reconnect: true,
      maxReconnectAttempts: 5,
      reconnectInterval: 2000,
    });
    crewConnections.set(key, client);
  }

  return crewConnections.get(key)!;
}

export function disconnectAllCrews(): void {
  crewConnections.forEach((client) => client.disconnect());
  crewConnections.clear();
}
