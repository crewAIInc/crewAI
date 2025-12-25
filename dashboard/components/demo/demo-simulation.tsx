"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useChatStore } from "@/stores/chat-store";
import { useTaskStore } from "@/stores/task-store";
import { useConnectionStore } from "@/stores/connection-store";
import { Play, Square, Zap, RotateCcw } from "lucide-react";
import type { StreamChunk } from "@/lib/types/stream";

// Sample agent roles
const agents = [
  { id: "cio-spot", role: "CIO Spot", desk: "spot" },
  { id: "research-head", role: "Research Head", desk: "spot" },
  { id: "systematic-head", role: "Systematic Head", desk: "spot" },
  { id: "execution-head", role: "Execution Head", desk: "spot" },
  { id: "risk-monitor", role: "Risk Monitor", desk: "spot" },
  { id: "gcro", role: "Group CRO", desk: "staff" },
  { id: "ceo", role: "CEO", desk: "staff" },
  { id: "carry-head", role: "Carry Head", desk: "futures" },
];

// Sample tasks
const sampleTasks = [
  "Market Analysis",
  "Portfolio Review",
  "Risk Assessment",
  "Order Execution",
  "Strategy Rebalance",
  "Exposure Check",
  "Funding Rate Analysis",
  "Compliance Verification",
];

// Sample messages
const sampleMessages = [
  "Initiating analysis of current market conditions...",
  "Reviewing portfolio allocations across all strategies.",
  "Checking risk exposure levels for all positions.",
  "Executing limit orders based on signal triggers.",
  "Rebalancing portfolio weights to match target allocation.",
  "Monitoring real-time exposure against limits.",
  "Analyzing funding rates across perpetual contracts.",
  "Verifying all trades comply with risk parameters.",
];

// Sample tool calls
const sampleTools = [
  { name: "GetAccountBalance", args: '{"asset": "all"}', result: '{"BTC": 2.5, "ETH": 45.2, "USDT": 150000}' },
  { name: "GetOpenOrders", args: '{}', result: '{"orders": [{"id": "123", "symbol": "BTCUSDT", "side": "buy"}]}' },
  { name: "GetTradeBalance", args: '{}', result: '{"equity": 425000, "margin": 12500, "free_margin": 412500}' },
  { name: "PlaceOrder", args: '{"symbol": "ETHUSDT", "side": "buy", "amount": 5}', result: '{"orderId": "456", "status": "filled"}' },
  { name: "GetTicker", args: '{"pair": "BTCUSD"}', result: '{"bid": 42150.5, "ask": 42151.2, "last": 42150.8}' },
];

let taskCounter = 0;
let messageCounter = 0;

export function DemoSimulation() {
  const [isRunning, setIsRunning] = useState(false);
  const [speed, setSpeed] = useState<"slow" | "normal" | "fast">("normal");
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const { addMessage, clearMessages } = useChatStore();
  const { addTask, updateTaskStatus, updateTaskProgress, completeTask } = useTaskStore();
  const { setConnected, setDisconnected } = useConnectionStore();

  const speedMs = { slow: 3000, normal: 1500, fast: 500 };

  const simulateEvent = useCallback(() => {
    const eventType = Math.random();

    if (eventType < 0.3) {
      // Start a new task
      const agent = agents[Math.floor(Math.random() * agents.length)];
      const taskName = sampleTasks[Math.floor(Math.random() * sampleTasks.length)];
      const taskId = `task-${++taskCounter}`;

      addTask({
        id: taskId,
        name: taskName,
        description: "",
        agentId: agent.id,
        agentRole: agent.role,
        status: "running",
        progress: 0,
        createdAt: new Date().toISOString(),
      });

      const chunk: StreamChunk = {
        content: sampleMessages[Math.floor(Math.random() * sampleMessages.length)],
        chunkType: "task_started",
        taskIndex: taskCounter,
        agentRole: agent.role,
        agentId: agent.id,
        taskName: taskName,
        taskId: taskId,
        timestamp: new Date().toISOString(),
      };
      addMessage(chunk, agent.desk as "staff" | "spot" | "futures");

    } else if (eventType < 0.5) {
      // Add a text message
      const agent = agents[Math.floor(Math.random() * agents.length)];
      const taskName = sampleTasks[Math.floor(Math.random() * sampleTasks.length)];

      const chunk: StreamChunk = {
        content: sampleMessages[Math.floor(Math.random() * sampleMessages.length)],
        chunkType: "text",
        taskIndex: 0,
        agentRole: agent.role,
        agentId: agent.id,
        taskName: taskName,
        taskId: `task-${taskCounter}`,
        timestamp: new Date().toISOString(),
      };
      addMessage(chunk, agent.desk as "staff" | "spot" | "futures");

    } else if (eventType < 0.7) {
      // Add a tool call
      const agent = agents[Math.floor(Math.random() * agents.length)];
      const tool = sampleTools[Math.floor(Math.random() * sampleTools.length)];
      const taskName = sampleTasks[Math.floor(Math.random() * sampleTasks.length)];

      const chunk: StreamChunk = {
        content: "",
        chunkType: "tool_call",
        taskIndex: 0,
        agentRole: agent.role,
        agentId: agent.id,
        taskName: taskName,
        taskId: `task-${taskCounter}`,
        timestamp: new Date().toISOString(),
        toolCall: {
          toolId: `tool-${++messageCounter}`,
          toolName: tool.name,
          arguments: tool.args,
          result: tool.result,
          success: true,
        },
      };
      addMessage(chunk, agent.desk as "staff" | "spot" | "futures");

    } else if (eventType < 0.85) {
      // Update task progress
      if (taskCounter > 0) {
        const taskId = `task-${Math.ceil(Math.random() * taskCounter)}`;
        const progress = Math.min(100, Math.floor(Math.random() * 40) + 20);
        updateTaskProgress(taskId, progress);
      }

    } else {
      // Complete a task
      if (taskCounter > 0) {
        const taskId = `task-${Math.ceil(Math.random() * taskCounter)}`;
        const duration = Math.random() * 5 + 0.5;
        completeTask(taskId, "Task completed successfully", duration);
      }
    }
  }, [addMessage, addTask, updateTaskProgress, completeTask]);

  const startSimulation = useCallback(() => {
    setIsRunning(true);
    setConnected();
    intervalRef.current = setInterval(simulateEvent, speedMs[speed]);
  }, [simulateEvent, speed, speedMs, setConnected]);

  const stopSimulation = useCallback(() => {
    setIsRunning(false);
    setDisconnected();
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, [setDisconnected]);

  const resetSimulation = useCallback(() => {
    stopSimulation();
    clearMessages();
    taskCounter = 0;
    messageCounter = 0;
  }, [stopSimulation, clearMessages]);

  // Update interval when speed changes
  useEffect(() => {
    if (isRunning && intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = setInterval(simulateEvent, speedMs[speed]);
    }
  }, [speed, isRunning, simulateEvent, speedMs]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  return (
    <Card className="p-4 bg-gradient-to-r from-primary/5 to-primary/10 border-primary/20">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Zap className="h-5 w-5 text-primary" />
          <div>
            <h3 className="text-sm font-medium">Demo Mode</h3>
            <p className="text-xs text-muted-foreground">
              Simulate agent activity without a server
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Speed selector */}
          <div className="flex items-center gap-1 bg-background rounded-lg p-1">
            {(["slow", "normal", "fast"] as const).map((s) => (
              <Button
                key={s}
                variant={speed === s ? "secondary" : "ghost"}
                size="sm"
                className="h-7 text-xs capitalize"
                onClick={() => setSpeed(s)}
              >
                {s}
              </Button>
            ))}
          </div>

          {/* Controls */}
          <Button
            variant={isRunning ? "destructive" : "default"}
            size="sm"
            onClick={isRunning ? stopSimulation : startSimulation}
            className="gap-2"
          >
            {isRunning ? (
              <>
                <Square className="h-3.5 w-3.5" />
                Stop
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5" />
                Start
              </>
            )}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={resetSimulation}
            className="gap-2"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </Button>
        </div>
      </div>

      {isRunning && (
        <div className="mt-3 flex items-center gap-2">
          <Badge variant="secondary" className="bg-green-500/10 text-green-500 text-xs">
            <span className="mr-1.5 h-1.5 w-1.5 rounded-full bg-green-500 animate-pulse" />
            Simulating
          </Badge>
          <span className="text-xs text-muted-foreground">
            Events every {speedMs[speed] / 1000}s
          </span>
        </div>
      )}
    </Card>
  );
}
