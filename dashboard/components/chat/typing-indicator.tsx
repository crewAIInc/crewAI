"use client";

import { cn } from "@/lib/utils";
import { AgentAvatar } from "@/components/agents/agent-avatar";
import type { Desk } from "@/lib/types/agent";

interface TypingIndicatorProps {
  agentName: string;
  desk: Desk;
  className?: string;
}

export function TypingIndicator({
  agentName,
  desk,
  className,
}: TypingIndicatorProps) {
  return (
    <div className={cn("flex items-start gap-3 animate-fade-in", className)}>
      <AgentAvatar name={agentName} desk={desk} size="md" />
      <div className="flex flex-col gap-1">
        <span className="text-xs font-medium text-muted-foreground">
          {agentName}
        </span>
        <div className="typing-indicator">
          <span />
          <span />
          <span />
        </div>
      </div>
    </div>
  );
}

interface MultiTypingIndicatorProps {
  agents: Array<{ name: string; desk: Desk }>;
  className?: string;
}

export function MultiTypingIndicator({
  agents,
  className,
}: MultiTypingIndicatorProps) {
  if (agents.length === 0) return null;

  if (agents.length === 1) {
    return (
      <TypingIndicator
        agentName={agents[0].name}
        desk={agents[0].desk}
        className={className}
      />
    );
  }

  const displayAgents = agents.slice(0, 3);
  const remainingCount = agents.length - 3;

  return (
    <div className={cn("flex items-center gap-3 animate-fade-in", className)}>
      <div className="flex -space-x-2">
        {displayAgents.map((agent, i) => (
          <AgentAvatar
            key={`${agent.name}-${i}`}
            name={agent.name}
            desk={agent.desk}
            size="sm"
            showTooltip={false}
            className="ring-2 ring-background"
          />
        ))}
      </div>
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">
          {displayAgents.map((a) => a.name).join(", ")}
          {remainingCount > 0 && ` +${remainingCount}`}
        </span>
        <div className="typing-indicator">
          <span />
          <span />
          <span />
        </div>
      </div>
    </div>
  );
}
