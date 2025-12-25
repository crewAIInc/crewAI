"use client";

import { useState, useRef, useEffect } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ChatTimeline } from "@/components/chat/chat-timeline";
import { ConnectionIndicator } from "@/components/layout/connection-status";
import { useWebSocket } from "@/components/providers/websocket-provider";
import { useChatStore, type Desk } from "@/stores/chat-store";
import { useTaskStore } from "@/stores/task-store";
import { AgentAvatar } from "@/components/agents/agent-avatar";
import {
  MessageSquare,
  Wifi,
  WifiOff,
  Trash2,
  Filter,
  Search,
  Send,
  Loader2,
  Users,
  Bot,
} from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const API_BASE = "http://localhost:8000";

interface ChatMessageType {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: string;
}

export default function ChatPage() {
  const [selectedCrew, setSelectedCrew] = useState<string>("staff");
  const [viewMode, setViewMode] = useState<"agents" | "chat">("agents");
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [chatHistory, setChatHistory] = useState<ChatMessageType[]>([]);
  const [isTyping, setIsTyping] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const wsUrl = selectedCrew === "all" ? null : `ws://localhost:8000/ws/${selectedCrew}`;
  const { connect, disconnect, isConnected, status } = useWebSocket();
  const { messages, clearMessages } = useChatStore();
  const { runningTaskIds, completedTaskIds, tasks } = useTaskStore();

  // Filter messages by selected crew
  const filteredMessages = selectedCrew === "all"
    ? messages
    : messages.filter((m) => m.desk === selectedCrew);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory, filteredMessages]);

  // Initial connection on mount
  useEffect(() => {
    if (selectedCrew !== "all" && !isConnected) {
      const url = `ws://localhost:8000/ws/${selectedCrew}`;
      connect(url);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Connect to WebSocket when crew changes (skip for "all")
  useEffect(() => {
    if (selectedCrew === "all") {
      // When "all" is selected, disconnect if connected
      if (isConnected) {
        disconnect();
      }
      return;
    }

    // Disconnect first if already connected
    if (isConnected) {
      disconnect();
    }

    // Connect to the selected crew's WebSocket
    const timer = setTimeout(() => {
      const url = `ws://localhost:8000/ws/${selectedCrew}`;
      connect(url);
    }, 200);

    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedCrew]);

  const handleConnect = () => {
    if (isConnected) {
      disconnect();
    } else if (wsUrl) {
      connect(wsUrl);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: ChatMessageType = {
      role: "user",
      content: inputMessage.trim(),
      timestamp: new Date().toISOString(),
    };

    // Add user message to history
    setChatHistory((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      setIsTyping(true);

      // Use REST API for chat (more reliable)
      const response = await fetch(`${API_BASE}/api/chat/${selectedCrew}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage.content,
          history: chatHistory,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const assistantMessage: ChatMessageType = {
          role: "assistant",
          content: data.response,
          timestamp: new Date().toISOString(),
        };
        setChatHistory((prev) => [...prev, assistantMessage]);
      } else {
        throw new Error("Server error");
      }
    } catch (error) {
      const errorMessage: ChatMessageType = {
        role: "assistant",
        content: `Fout: ${error instanceof Error ? error.message : "Kon geen verbinding maken"}`,
        timestamp: new Date().toISOString(),
      };
      setChatHistory((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleClearChat = async () => {
    setChatHistory([]);
    clearMessages();
    try {
      await fetch(`${API_BASE}/api/chat/${selectedCrew}/clear`, {
        method: "POST",
      });
    } catch (error) {
      console.error("Failed to clear server session:", error);
    }
  };

  const getCrewColor = (crew: string) => {
    switch (crew) {
      case "all":
        return "bg-primary/20 text-primary border-primary/50";
      case "staff":
        return "bg-purple-500/20 text-purple-400 border-purple-500/50";
      case "spot":
        return "bg-orange-500/20 text-orange-400 border-orange-500/50";
      case "futures":
        return "bg-teal-500/20 text-teal-400 border-teal-500/50";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const getCrewName = (crew: string) => {
    switch (crew) {
      case "all":
        return "Alle Crews";
      case "staff":
        return "STAFF (Executive Board)";
      case "spot":
        return "Spot Trading Desk";
      case "futures":
        return "Futures Trading Desk";
      default:
        return crew;
    }
  };

  const getDeskColor = (desk: Desk) => {
    switch (desk) {
      case "staff":
        return "bg-purple-500";
      case "spot":
        return "bg-orange-500";
      case "futures":
        return "bg-teal-500";
    }
  };

  const messageCount = messages.length;
  const runningTasks = runningTaskIds.map((id) => tasks.get(id)).filter(Boolean);
  const completedCount = completedTaskIds.length;

  return (
    <div className="h-[calc(100vh-8rem)] flex gap-6 animate-fade-in">
      {/* Chat Panel */}
      <Card className="flex-1 flex flex-col overflow-hidden glass-card">
        <CardHeader className="flex-shrink-0 border-b border-white/5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5 text-primary" />
                Chat met Crew
              </CardTitle>
              <ConnectionIndicator />
            </div>
            <div className="flex items-center gap-2">
              {/* Crew Selector */}
              <Select value={selectedCrew} onValueChange={setSelectedCrew}>
                <SelectTrigger className={`w-48 h-9 border ${getCrewColor(selectedCrew)}`}>
                  <Users className="h-4 w-4 mr-2" />
                  <SelectValue placeholder="Selecteer crew" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-primary" />
                      Alle Crews (74 agents)
                    </div>
                  </SelectItem>
                  <SelectItem value="staff">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-purple-500" />
                      STAFF (10 agents)
                    </div>
                  </SelectItem>
                  <SelectItem value="spot">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-orange-500" />
                      Spot Desk (32 agents)
                    </div>
                  </SelectItem>
                  <SelectItem value="futures">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-teal-500" />
                      Futures Desk (32 agents)
                    </div>
                  </SelectItem>
                </SelectContent>
              </Select>

              <Button
                variant="ghost"
                size="icon"
                onClick={handleClearChat}
                className="text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        {/* Connection Status */}
        <div className="flex items-center gap-3 px-4 py-2 border-b border-white/5 bg-muted/30">
          <div className="flex-1 flex items-center gap-2">
            <Badge
              variant="outline"
              className={`text-xs ${
                isConnected
                  ? "border-green-500/50 text-green-400"
                  : "border-red-500/50 text-red-400"
              }`}
            >
              {isConnected ? (
                <>
                  <Wifi className="h-3 w-3 mr-1" />
                  Verbonden
                </>
              ) : (
                <>
                  <WifiOff className="h-3 w-3 mr-1" />
                  Niet verbonden
                </>
              )}
            </Badge>
            <span className="text-xs text-muted-foreground">
              {getCrewName(selectedCrew)}
            </span>
          </div>
          <Button
            onClick={handleConnect}
            variant={isConnected ? "destructive" : "default"}
            size="sm"
            className="gap-2 h-7 text-xs"
          >
            {isConnected ? "Disconnect" : "Connect"}
          </Button>
        </div>

        {/* Agent Messages */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-3">
            {filteredMessages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-center">
                <Bot className="h-16 w-16 text-muted-foreground/30 mb-4" />
                <h3 className="text-lg font-medium text-muted-foreground">
                  Geen agent activiteit
                </h3>
                <p className="text-sm text-muted-foreground/60 max-w-md mt-2">
                  Start een crew via de Overview pagina om agent conversaties te zien.
                  {selectedCrew !== "all" && " Of selecteer 'Alle Crews' om alle activiteit te zien."}
                </p>
              </div>
            ) : (
              filteredMessages.map((msg) => (
                <div
                  key={msg.id}
                  className="group rounded-xl border border-white/10 bg-gradient-to-br from-muted/30 to-muted/10 p-4 transition-all duration-300 hover:border-white/20 hover:shadow-lg hover:shadow-primary/5"
                >
                  {/* Agent Header with Crew Color */}
                  <div className="flex items-start gap-3 mb-3">
                    <div className="relative flex-shrink-0">
                      <div className={`absolute -inset-1 rounded-full ${getDeskColor(msg.desk)} opacity-20 blur-sm`} />
                      <AgentAvatar name={msg.agentRole} desk={msg.desk} size="sm" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-sm font-semibold text-foreground leading-tight">
                          {msg.agentRole}
                        </span>
                        <div className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase tracking-wider ${
                          msg.desk === 'staff' ? 'bg-purple-500/20 text-purple-400' :
                          msg.desk === 'spot' ? 'bg-orange-500/20 text-orange-400' :
                          'bg-teal-500/20 text-teal-400'
                        }`}>
                          {msg.desk}
                        </div>
                      </div>
                      {msg.taskName && (
                        <Badge variant="outline" className="text-[9px] mt-1.5 border-primary/30 text-primary/80">
                          {msg.taskName}
                        </Badge>
                      )}
                    </div>
                    <span className="text-[10px] text-muted-foreground tabular-nums flex-shrink-0">
                      {new Date(msg.timestamp).toLocaleTimeString("nl-NL")}
                    </span>
                  </div>

                  {/* Message Content - Full Height, Always Visible */}
                  <div className="pl-10">
                    <p className="text-sm text-foreground/90 leading-relaxed whitespace-pre-wrap break-words">
                      {msg.content}
                    </p>

                    {/* Tool Calls */}
                    {msg.toolCalls.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {msg.toolCalls.map((tool, idx) => (
                          <Badge
                            key={idx}
                            variant="secondary"
                            className="text-[10px] bg-blue-500/15 text-blue-400 border border-blue-500/30 font-medium"
                          >
                            <span className="mr-1">🔧</span>
                            {tool.toolName}
                          </Badge>
                        ))}
                      </div>
                    )}

                    {/* Streaming Indicator */}
                    {msg.isStreaming && (
                      <div className="flex items-center gap-2 mt-3 py-1.5 px-2.5 rounded-lg bg-primary/10 border border-primary/20 w-fit">
                        <div className="flex gap-1">
                          <div className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                          <div className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                          <div className="w-1.5 h-1.5 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                        </div>
                        <span className="text-[10px] text-primary font-medium">Agent is typing...</span>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}

            <div ref={chatEndRef} />
          </div>
        </ScrollArea>

        {/* Chat Input */}
        <div className="flex-shrink-0 border-t border-white/5 p-4 bg-background/50">
          <div className="flex items-center gap-3">
            <Input
              placeholder={`Bericht aan ${selectedCrew.toUpperCase()} crew...`}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyPress}
              disabled={isLoading}
              className="flex-1 h-11 bg-muted/30 border-white/10 focus:border-primary/50"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading}
              size="icon"
              className="h-11 w-11 shrink-0"
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                <Send className="h-5 w-5" />
              )}
            </Button>
          </div>
          <p className="text-xs text-muted-foreground mt-2">
            Tip: Vraag om &quot;analyseer BTC markt&quot; of &quot;wat is de huidige funding rate?&quot;
          </p>
        </div>
      </Card>

      {/* Right Panel: Active Tasks & Info */}
      <Card className="w-80 flex-shrink-0 glass-card">
        <CardHeader className="border-b border-white/5">
          <CardTitle className="text-sm flex items-center justify-between">
            Crew Info
            <Badge variant="secondary" className={`text-[10px] ${getCrewColor(selectedCrew)}`}>
              {selectedCrew === "staff" ? "10" : "32"} agents
            </Badge>
          </CardTitle>
        </CardHeader>
        <ScrollArea className="h-[calc(100vh-14rem)]">
          <div className="p-4 space-y-4">
            {/* Crew Description */}
            <div className="rounded-lg border border-white/5 p-3 bg-muted/20">
              <h4 className="font-medium text-sm mb-2">{getCrewName(selectedCrew)}</h4>
              <p className="text-xs text-muted-foreground">
                {selectedCrew === "staff" &&
                  "C-level executives: CEO, CIO, CRO, COO, CFO. Verantwoordelijk voor governance, strategie en risicobeheer."}
                {selectedCrew === "spot" &&
                  "Spot trading operaties: Systematic, Discretionary, Arbitrage, Research, Execution en Market Making teams."}
                {selectedCrew === "futures" &&
                  "Futures/derivatives trading: Carry trades, Microstructure, Swing trading, Risk monitoring en Treasury management."}
              </p>
            </div>

            <Separator className="bg-white/5" />

            {/* Active Tasks */}
            <div>
              <h4 className="text-xs font-medium text-muted-foreground mb-3 flex items-center justify-between">
                Active Tasks
                <Badge variant="secondary" className="text-[10px]">
                  {runningTasks.length} running
                </Badge>
              </h4>
              {runningTasks.length === 0 ? (
                <div className="text-center text-sm text-muted-foreground py-4">
                  <p className="text-xs">Geen actieve taken</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {runningTasks.map(
                    (task) =>
                      task && (
                        <div
                          key={task.id}
                          className="rounded-lg border border-blue-500/30 p-2 bg-blue-500/5"
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-medium truncate">
                              {task.name}
                            </span>
                            <Badge className="status-running text-[9px] px-1.5 py-0">
                              Running
                            </Badge>
                          </div>
                          <div className="flex items-center gap-2 mt-1">
                            <AgentAvatar
                              name={task.agentRole}
                              desk={selectedCrew as "spot" | "futures" | "staff"}
                              size="sm"
                            />
                            <span className="text-[10px] text-muted-foreground truncate">
                              {task.agentRole}
                            </span>
                          </div>
                        </div>
                      )
                  )}
                </div>
              )}
            </div>

            {completedCount > 0 && (
              <>
                <Separator className="bg-white/5" />
                <div className="text-xs text-muted-foreground text-center py-2">
                  {completedCount} taken voltooid
                </div>
              </>
            )}

            <Separator className="bg-white/5" />

            {/* Quick Actions */}
            <div>
              <h4 className="text-xs font-medium text-muted-foreground mb-3">
                Suggesties
              </h4>
              <div className="space-y-2">
                {[
                  "Analyseer de huidige markt",
                  "Wat is de BTC prijs?",
                  "Geef een risk rapport",
                  "Start monitoring",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInputMessage(suggestion)}
                    className="w-full text-left text-xs px-3 py-2 rounded-lg border border-white/5 bg-muted/20 hover:bg-muted/40 transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </ScrollArea>
      </Card>
    </div>
  );
}
