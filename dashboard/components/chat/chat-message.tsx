"use client";

import { cn } from "@/lib/utils";
import { AgentAvatar } from "@/components/agents/agent-avatar";
import { ToolCallCard } from "@/components/chat/tool-call-card";
import { Badge } from "@/components/ui/badge";
import { formatDistanceToNow } from "date-fns";
import type { ChatMessage as ChatMessageType } from "@/stores/chat-store";
import type { Desk } from "@/lib/types/agent";

interface ChatMessageProps {
  message: ChatMessageType;
  desk: Desk;
  showAvatar?: boolean;
  className?: string;
}

export function ChatMessage({
  message,
  desk,
  showAvatar = true,
  className,
}: ChatMessageProps) {
  const timeAgo = formatDistanceToNow(new Date(message.timestamp), {
    addSuffix: true,
  });

  return (
    <div
      className={cn(
        "group flex gap-3 px-4 py-3 transition-colors animate-slide-up",
        "hover:bg-muted/30",
        className
      )}
    >
      {/* Avatar */}
      <div className="flex-shrink-0 pt-0.5">
        {showAvatar ? (
          <AgentAvatar name={message.agentRole} desk={desk} size="md" />
        ) : (
          <div className="w-8" />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0 space-y-1.5">
        {/* Header */}
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm">{message.agentRole}</span>
          <Badge
            variant="secondary"
            className={cn(
              "text-[10px] px-1.5 py-0",
              desk === "staff" && "desk-staff",
              desk === "spot" && "desk-spot",
              desk === "futures" && "desk-futures"
            )}
          >
            {desk.toUpperCase()}
          </Badge>
          {message.taskName && (
            <Badge variant="outline" className="text-[10px] px-1.5 py-0">
              {message.taskName}
            </Badge>
          )}
          <span className="text-xs text-muted-foreground ml-auto opacity-0 group-hover:opacity-100 transition-opacity">
            {timeAgo}
          </span>
        </div>

        {/* Message Content */}
        {message.content && (
          <div className="prose prose-sm prose-neutral dark:prose-invert max-w-none">
            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {message.content}
              {message.isStreaming && (
                <span className="inline-block w-2 h-4 ml-0.5 bg-primary animate-pulse" />
              )}
            </p>
          </div>
        )}

        {/* Tool Calls */}
        {message.toolCalls.length > 0 && (
          <div className="space-y-2 mt-2">
            {message.toolCalls.map((toolCall, index) => (
              <ToolCallCard
                key={`${toolCall.toolId}-${index}`}
                toolCall={toolCall}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// Compact message for when same agent sends multiple messages
interface CompactChatMessageProps {
  message: ChatMessageType;
  className?: string;
}

export function CompactChatMessage({
  message,
  className,
}: CompactChatMessageProps) {
  return (
    <div
      className={cn(
        "group flex gap-3 px-4 py-1 transition-colors animate-fade-in",
        "hover:bg-muted/30",
        className
      )}
    >
      {/* Spacer for avatar alignment */}
      <div className="w-8 flex-shrink-0" />

      {/* Content */}
      <div className="flex-1 min-w-0">
        {message.content && (
          <p className="text-sm leading-relaxed whitespace-pre-wrap">
            {message.content}
            {message.isStreaming && (
              <span className="inline-block w-2 h-4 ml-0.5 bg-primary animate-pulse" />
            )}
          </p>
        )}

        {/* Tool Calls */}
        {message.toolCalls.length > 0 && (
          <div className="space-y-2 mt-2">
            {message.toolCalls.map((toolCall, index) => (
              <ToolCallCard
                key={`${toolCall.toolId}-${index}`}
                toolCall={toolCall}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
